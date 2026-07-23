// SPDX-License-Identifier: LGPL-3.0-or-later

#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

#include "sockets.h"
#include "sockets_internal.h"

#define CHECK(condition)                                                 \
  do {                                                                   \
    if (!(condition)) {                                                  \
      fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, \
              #condition);                                               \
      return 1;                                                          \
    }                                                                    \
  } while (0)

static char received[32];
static size_t received_size;
static int write_calls;

static ssize_t partial_writer(int sockfd, const void* data, size_t len) {
  size_t count;
  (void)sockfd;

  ++write_calls;
  if (write_calls == 1) {
    // Signals may interrupt write() before it makes progress.  The complete
    // write helper must retry without advancing the source pointer.
    errno = EINTR;
    return -1;
  }

  // Deliberately accept at most three bytes so the test cannot accidentally
  // pass with the historical one-shot writebuffer_ implementation.
  count = len < 3 ? len : 3;
  memcpy(received + received_size, data, count);
  received_size += count;
  return (ssize_t)count;
}

static ssize_t zero_writer(int sockfd, const void* data, size_t len) {
  (void)sockfd;
  (void)data;
  (void)len;
  return 0;
}

static ssize_t error_writer(int sockfd, const void* data, size_t len) {
  (void)sockfd;
  (void)data;
  (void)len;
  errno = ECONNRESET;
  return -1;
}

static ssize_t overreporting_writer(int sockfd,
                                    const void* data,
                                    size_t len) {
  (void)sockfd;
  (void)data;
  return (ssize_t)(len + 1);
}

static int test_unix_socket_path_bounds(void) {
  struct sockaddr_un address;
  const size_t prefix_length = strlen("/tmp/ipi_");
  const size_t max_host_length = sizeof(address.sun_path) - prefix_length - 1;
  char host[sizeof(address.sun_path) + 1];

  memset(host, 'a', max_host_length);
  host[max_host_length] = '\0';

  errno = 0;
  CHECK(deepmd_build_unix_socket_address(NULL, host) == -1);
  CHECK(errno == EINVAL);
  errno = 0;
  CHECK(deepmd_build_unix_socket_address(&address, NULL) == -1);
  CHECK(errno == EINVAL);

  CHECK(deepmd_build_unix_socket_address(&address, host) == 0);
  CHECK(address.sun_family == AF_UNIX);
  CHECK(strlen(address.sun_path) == sizeof(address.sun_path) - 1);
  CHECK(strcmp(address.sun_path + prefix_length, host) == 0);

  host[max_host_length] = 'b';
  host[max_host_length + 1] = '\0';
  errno = 0;
  CHECK(deepmd_build_unix_socket_address(&address, host) == -1);
  CHECK(errno == ENAMETOOLONG);
  return 0;
}

static int test_complete_writes(void) {
  static const char payload[] = "partial-write-payload";

  errno = 0;
  CHECK(deepmd_write_all(123, payload, sizeof(payload) - 1, NULL) == -1);
  CHECK(errno == EINVAL);
  errno = 0;
  CHECK(deepmd_write_all(123, NULL, 1, partial_writer) == -1);
  CHECK(errno == EINVAL);

  memset(received, 0, sizeof(received));
  received_size = 0;
  write_calls = 0;
  CHECK(deepmd_write_all(123, payload, sizeof(payload) - 1, partial_writer) ==
        0);
  CHECK(received_size == sizeof(payload) - 1);
  CHECK(memcmp(received, payload, sizeof(payload) - 1) == 0);
  CHECK(write_calls > 2);

  errno = 0;
  CHECK(deepmd_write_all(123, payload, sizeof(payload) - 1, zero_writer) == -1);
  CHECK(errno == EPIPE);

  errno = 0;
  CHECK(deepmd_write_all(123, payload, sizeof(payload) - 1, error_writer) ==
        -1);
  CHECK(errno == ECONNRESET);

  errno = 0;
  CHECK(deepmd_write_all(123, payload, sizeof(payload) - 1,
                         overreporting_writer) == -1);
  CHECK(errno == EIO);
  return 0;
}

static int test_writebuffer_rejects_negative_length(void) {
  char error_output[256] = {0};
  char payload = 'x';
  int error_pipe[2];
  int sockfd = -1;
  int status;
  pid_t child;
  ssize_t count;

  CHECK(pipe(error_pipe) == 0);
  child = fork();
  CHECK(child >= 0);
  if (child == 0) {
    close(error_pipe[0]);
    if (dup2(error_pipe[1], STDERR_FILENO) < 0) {
      _exit(2);
    }
    close(error_pipe[1]);
    writebuffer_(&sockfd, &payload, -1);
    _exit(0);
  }

  close(error_pipe[1]);
  count = read(error_pipe[0], error_output, sizeof(error_output) - 1);
  close(error_pipe[0]);
  CHECK(count > 0);
  CHECK(waitpid(child, &status, 0) == child);
  CHECK(WIFEXITED(status));
  CHECK(WEXITSTATUS(status) == 255);
  CHECK(strstr(error_output, "invalid buffer length") != NULL);
  return 0;
}

int main(void) {
  if (test_unix_socket_path_bounds() != 0) {
    return 1;
  }
  if (test_complete_writes() != 0) {
    return 1;
  }
  if (test_writebuffer_rejects_negative_length() != 0) {
    return 1;
  }
  return 0;
}
