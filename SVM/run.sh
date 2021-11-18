#!/bin/sh
echo " "
echo "Jason Zavala proudly presents:"
echo "HW4"
echo "Section 2"
echo " "
echo "###########################################################"
echo "                       PART 2A"
echo "###########################################################"
echo " "

echo "C = 100/873"
python3 svm.py 0.114 0
echo "C = 500/873"
python3 svm.py 0.573 0
echo "C = 700/873"
python3 svm.py .802 0

echo " "
echo "###########################################################"
echo "                       PART 2b"
echo "###########################################################"
echo " "
echo "C = 100/873"
python3 svm.py 0.114 1
echo "C = 500/873"
python3 svm.py 0.573 1
echo "C = 700/873"
python3 svm.py .802 1

echo " "
echo "###########################################################"
echo "                       PART 3"
echo "###########################################################"
