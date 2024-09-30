#!/usr/bin/env bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
mkdir -p checkpoints
# Place final weights here:
wget https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt -P checkpoints
