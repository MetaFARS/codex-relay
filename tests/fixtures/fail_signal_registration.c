#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>

/* This library is loaded into a regression-test child, never the test runner. */
int sigaction(int number, const struct sigaction *action, struct sigaction *old) {
    const char *fault = getenv("RELAY_SIGNAL_FAULT");
    if (action && fault &&
        ((number == SIGINT && strcmp(fault, "INT") == 0) ||
         (number == SIGTERM && strcmp(fault, "TERM") == 0))) {
        errno = EPERM;
        return -1;
    }
    int (*original)(int, const struct sigaction *, struct sigaction *) =
        dlsym(RTLD_NEXT, "sigaction");
    return original(number, action, old);
}
