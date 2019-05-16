#!/bin/sh

mkdir Test1
cd Test1/
smbclient -U% //51.15.62.219/sambashare -c "cd Test1/;prompt;mget *"
cd ../
mkdir Train1
smbclient -U% //51.15.62.219/sambashare -c "cd Train1/;prompt;mget *"
cd ../
mkdir Train
smbclient -U% //51.15.62.219/sambashare -c "cd Train/;prompt;mget *"
