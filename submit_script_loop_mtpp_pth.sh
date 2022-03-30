#!/bin/bash

PRINTDIR="/home/amuntoni/sibyl-team/loop_mtpp/"
beta=0.05
ti=$1
fn_rate=$2
adp_frac=$3
prob_th=$4
tau=$5

for alg in bp_gamma_th
do
	for seed in $(seq 1 1 11) 
	do
		for nobs in 400
		do
			script=$(echo "script_loop_mtpp_"$alg".py")
			flag=$seed"_"$alg"_"$nobs"_"$ti"_fnr_"$fn_rate"_af_"$adp_frac"_pth_"$prob_th"_tau_"$tau
			echo $flag
			python3  $script "--fn_rate" $fn_rate "--tau" $tau "-m" $seed "-p" $prob_th "-n" $nobs "-b" $beta "-i" $ti "--adp_frac" $adp_frac 	
			
		done
	done
done
