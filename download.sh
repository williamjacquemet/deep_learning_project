#!/bin/sh

sudo apt-get update
sudo apt-get -y install smbclient
smbclient -U% //51.15.62.219/sambashare -c "get f.pickle train.csv test.csv"
