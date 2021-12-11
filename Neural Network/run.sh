#!/bin/sh
echo " "
echo "Jason Zavala proudly presents:"
echo "HW5"
echo "Section 2"
echo " "
echo "###########################################################"
echo "                       PART A"
echo "###########################################################"
echo " "

echo "Back propagation"
python3 nn.py part_a
echo " "

echo " "
echo "###########################################################"
echo "                       PART B"
echo "###########################################################"
echo " "
echo "Stochastic Gradient Descent"
python3 nn.py part_b

echo " "
echo "###########################################################"
echo "                       PART C"
echo "###########################################################"
echo " "
echo "Weights initialized to 0"
python3 nn.py part_c