#!/bin/bash


SYCL_PI_TRACE=-1 ZE_DEBUG=1 SYCL_DEVICE_FILTER=level_zero:gpu:0.0 bin/fft-test --gtest_filter=S\*Dev\*/4_4\* 2>&1| tee debug.txt
