//go:build linux

package main

import (
	"syscall"
	"time"
)

func threadCPU() (time.Duration, bool) {
	var ru syscall.Rusage
	if err := syscall.Getrusage(syscall.RUSAGE_THREAD, &ru); err != nil {
		return 0, false
	}
	return timevalDuration(ru.Utime) + timevalDuration(ru.Stime), true
}

func timevalDuration(tv syscall.Timeval) time.Duration {
	return time.Duration(tv.Sec)*time.Second + time.Duration(tv.Usec)*time.Microsecond
}

func dutyClockName() string {
	return "thread-CPU (RUSAGE_THREAD)"
}
