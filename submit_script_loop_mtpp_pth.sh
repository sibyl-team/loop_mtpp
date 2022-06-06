#!/bin/bash

PRINTDIR="/home/amuntoni/sibyl-team/loop_mtpp/"
beta=0.05
ti=$1
fn_rate=0.285
quarHH=1
testHH=0
test_rnd=0
frac_sym=0.8
T=$2
adp_frac=$3
tau=$4
prob_th=0.0005


for seed in $(seq 6 1 10)
do
    for alg in bp_gamma_th mf_th
	do			
		for nobs in 400
		do
            script=$(echo "script_loop_mtpp_"$alg".py")
			flag=$seed"_"$alg"_"$nobs"_"$ti"_fnr_"$fn_rate"_af_"$adp_frac"_pth_"$prob_th"_tau_"$tau
			echo $flag
			python3  $script "--test_HH" $testHH "--frac_sym" $frac_sym "--fn_rate" $fn_rate "--tau" $tau "-m" $seed  "-n" $nobs "-b" $beta "-i" $ti "--adp_frac" $adp_frac "--quarantine_HH" $quarHH "--or" $test_rnd "-T" $T "--out" "output_Tubingen_pop1_site1_fnr" "-p" $prob_th
		done
	done
done


