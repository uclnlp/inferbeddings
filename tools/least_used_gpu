#!/bin/bash

nvidia-smi dmon -c 1 | grep -v "#" | awk '{ print $4 " " $1 }' | sort -n | awk '{ print $2 }' | tr -d "\n" | head -c 1
