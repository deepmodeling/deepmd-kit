// SPDX-License-Identifier: LGPL-3.0-or-later
/* A minimal wrapper for socket communication.

Copyright (C) 2013, Joshua More and Michele Ceriotti

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


Contains both the functions that transmit data to the socket and read the data
back out again once finished, and the function which opens the socket initially.
Can be linked to a FORTRAN code that does not support sockets natively.

Functions:
   error: Prints an error message and then exits.
   open_socket_: Opens a socket with the required host server, socket type and
      port number.
   write_buffer_: Writes a string to the socket.
   read_buffer_: Reads data from the socket.
*/

#include <errno.h>
#include <netdb.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>
#include <unistd.h>

#include "sockets_internal.h"

int deepmd_build_unix_socket_address(struct sockaddr_un* address,
                                     const char* host) {
  static const char prefix[] = "/tmp/ipi_";
  size_t host_length;

  if (address == NULL || host == NULL) {
    errno = EINVAL;
    return -1;
  }

  host_length = strlen(host);
  // sizeof(prefix) includes its terminator, exactly reserving the byte needed
  // after the host.  Rejecting the name is safer than silently connecting to
  // a truncated socket or overflowing sockaddr_un::sun_path.
  if (host_length > sizeof(address->sun_path) - sizeof(prefix)) {
    errno = ENAMETOOLONG;
    return -1;
  }

  memset(address, 0, sizeof(*address));
  address->sun_family = AF_UNIX;
  memcpy(address->sun_path, prefix, sizeof(prefix) - 1);
  memcpy(address->sun_path + sizeof(prefix) - 1, host, host_length + 1);
  return 0;
}

int deepmd_write_all(int sockfd,
                     const char* data,
                     size_t len,
                     deepmd_socket_write_fn write_fn) {
  size_t written = 0;

  if (write_fn == NULL || (data == NULL && len != 0)) {
    errno = EINVAL;
    return -1;
  }

  while (written < len) {
    ssize_t count = write_fn(sockfd, data + written, len - written);
    if (count > 0) {
      // A conforming write() cannot report more bytes than requested.  Keep
      // this guard because tests and alternative wrappers can supply the
      // callback, and advancing past len would turn their bug into an OOB
      // pointer on the next iteration.
      if ((size_t)count > len - written) {
        errno = EIO;
        return -1;
      }
      written += (size_t)count;
    } else if (count == 0) {
      // A zero-length progress report for a nonempty request would otherwise
      // spin forever.  Stream peers that stop accepting data are treated as a
      // broken connection, matching the public writebuffer_ contract.
      errno = EPIPE;
      return -1;
    } else if (errno != EINTR) {
      return -1;
    }
  }
  return 0;
}

void error(const char* msg)
// Prints an error message and then exits.
{
  perror(msg);
  exit(-1);
}

void open_socket_(int* psockfd, int* inet, int* port, const char* host)
/* Opens a socket.

Note that fortran passes an extra argument for the string length, but this is
ignored here for C compatibility.

Args:
   psockfd: The id of the socket that will be created.
   inet: An integer that determines whether the socket will be an inet or unix
      domain socket. Gives unix if 0, inet otherwise.
   port: The port number for the socket to be created. Low numbers are often
      reserved for important channels, so use of numbers of 4 or more digits is
      recommended.
   host: The name of the host server.
*/

{
  int sockfd;
  struct hostent* server;

  struct sockaddr* psock;
  int ssock;

  if (*inet > 0) {  // creates an internet socket
    struct sockaddr_in serv_addr;
    psock = (struct sockaddr*)&serv_addr;
    ssock = sizeof(serv_addr);
    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) {
      error("Error opening socket");
    }

    server = gethostbyname(host);
    if (server == NULL) {
      fprintf(stderr, "Error opening socket: no such host %s \n", host);
      exit(-1);
    }

    bzero((char*)&serv_addr, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    bcopy((char*)server->h_addr, (char*)&serv_addr.sin_addr.s_addr,
          server->h_length);
    serv_addr.sin_port = htons(*port);
    if (connect(sockfd, psock, ssock) < 0) {
      error("Error opening socket: wrong host address, or broken connection");
    }
  } else {  // creates a unix socket
    struct sockaddr_un serv_addr;
    psock = (struct sockaddr*)&serv_addr;
    ssock = sizeof(serv_addr);
    if (deepmd_build_unix_socket_address(&serv_addr, host) < 0) {
      error("Error opening socket: Unix socket path is too long");
    }
    sockfd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (sockfd < 0) {
      error("Error opening socket");
    }
    if (connect(sockfd, psock, ssock) < 0) {
      error("Error opening socket: wrong host address, or broken connection");
    }
  }

  *psockfd = sockfd;
}

void writebuffer_(int* psockfd, char* data, int len)
/* Writes to a socket.

Args:
   psockfd: The id of the socket that will be written to.
   data: The data to be written to the socket.
   plen: The length of the data in bytes.
*/

{
  int sockfd = *psockfd;

  if (len < 0) {
    errno = EINVAL;
    error("Error writing to socket: invalid buffer length");
  }
  if (deepmd_write_all(sockfd, data, (size_t)len, write) < 0) {
    error("Error writing to socket: server has quit or connection broke");
  }
}

void readbuffer_(int* psockfd, char* data, int len)
/* Reads from a socket.

Args:
   psockfd: The id of the socket that will be read from.
   data: The storage array for data read from the socket.
   plen: The length of the data in bytes.
*/

{
  int n, nr;
  int sockfd = *psockfd;

  n = nr = read(sockfd, data, len);

  while (nr > 0 && n < len) {
    nr = read(sockfd, &data[n], len - n);
    n += nr;
  }

  if (n == 0) {
    error("Error reading from socket: server has quit or connection broke");
  }
}
