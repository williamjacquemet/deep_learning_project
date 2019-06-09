#!/bin/sh

sudo apt-get update
sudo apt-get -y install smbclient
smbclient -U% //51.158.187.148/sambashare -c "get f.pickle train.csv test.csv"
