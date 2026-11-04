#!/bin/sh
sudo systemctl stop bruhos-web bruhos-ear joystick multi_control_client multi_control_server tonypi

sudo fuser /dev/ttyAMA0
