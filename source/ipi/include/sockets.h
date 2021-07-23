#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <sys/un.h>
#include <netdb.h>

#ifdef __cplusplus
extern "C" {
#endif

    void error (const char *msg);

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
    void open_socket_(int *psockfd, int* inet, int* port, const char* host);

/* Writes to a socket.
   Args:
   psockfd: The id of the socket that will be written to.
   data: The data to be written to the socket.
   plen: The length of the data in bytes.
*/
    void writebuffer_(int *psockfd, char *data, int len);    

/* Reads from a socket.
   Args:
   psockfd: The id of the socket that will be read from.
   data: The storage array for data read from the socket.
   plen: The length of the data in bytes.
*/
    void readbuffer_(int *psockfd, char *data, int len);    
    
#ifdef __cplusplus
}
#endif
