#!/bin/bash

fea_dim=$1

splice="$2"

len=`echo $splice | wc -w`

NN_dim=$(( len * fea_dim ))

echo "<splice> $NN_dim $fea_dim"
echo -n "[ "
 for i in "${splice[@]}"
 do
   echo -n "$i "
 done

echo "]" 
