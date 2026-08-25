package main

import (
	"time"
)

type workSpan struct {
	cpu0  time.Duration
	wall0 time.Time
	cpuOK bool
}

func startWork() workSpan {
	c, ok := threadCPU()
	return workSpan{cpu0: c, wall0: time.Now(), cpuOK: ok}
}

func (w workSpan) elapsed() time.Duration {
	if w.cpuOK {
		c, ok := threadCPU()
		if ok && c >= w.cpu0 {
			return c - w.cpu0
		}
	}
	return time.Since(w.wall0)
}
