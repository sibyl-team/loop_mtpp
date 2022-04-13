#!/bin/bash

PRINTDIR="/home/amuntoni/sibyl-team/loop_mtpp/"
beta=0.05
ti=$1
fn_rate=$2
adp_frac=$3
tau=$4
#prob_th=0.001


for seed in $(seq 1 1 9)
do
    for alg in bp_gamma
	do			
		for nobs in 400
		do
			script=$(echo "script_loop_mtpp_"$alg".py")
			flag=$seed"_"$alg"_"$nobs"_"$ti"_fnr_"$fn_rate"_af_"$adp_frac"_tau_"$tau
			echo $flag
			python3  $script "--fn_rate" $fn_rate "--tau" $tau "-m" $seed  "-n" $nobs "-b" $beta "-i" $ti "--adp_frac" $adp_frac 
		done
	done
done
