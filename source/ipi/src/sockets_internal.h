// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <stddef.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>

/* Internal, testable primitives used by the public Fortran-compatible API. */

typedef ssize_t (*deepmd_socket_write_fn)(int, const void*, size_t);

/* Build /tmp/ipi_<host> without truncating or overflowing sun_path. */
int deepmd_build_unix_socket_address(struct sockaddr_un* address,
                                     const char* host);

/* Retry interrupted and partial writes until len bytes have been sent. */
int deepmd_write_all(int sockfd,
                     const char* data,
                     size_t len,
                     deepmd_socket_write_fn write_fn);
