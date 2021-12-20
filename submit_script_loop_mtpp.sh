#!/bin/bash

PRINTDIR="/home/amuntoni/sibyl-team/loop_mtpp/"
beta=0.05
ti=$1
fn_rate=$2
adp_frac=$3
tau=$4
#prob_th=0.001

for alg in bp_gamma
do
	for seed in $(seq 1 1 10) 
	#for seed in 2
	do
		#for nobs in 60 90 120 150
		for nobs in 400
		do
			script=$(echo "script_loop_mtpp_"$alg".py")
			#flag=$seed"_"$alg"_"$nobs"_fnr_"$fn_rate"_af_"$adp_frac"_pth_"$prob_th
			flag=$seed"_"$alg"_"$nobs"_"$ti"_fnr_"$fn_rate"_af_"$adp_frac"_tau_"$tau
			echo "#!/bin/bash" > slurm_$flag".sh"
			echo " " >> slurm_$flag".sh"
			echo "#SBATCH --job-name="$flag >> slurm_$flag".sh"
			echo "#SBATCH --partition=global" >> slurm_$flag".sh"
			echo "#SBATCH --time=240:00:00" >> slurm_$flag".sh"
			if [ "$alg" == "bp_exp" ] || [ "$alg" == "bp_gamma" ]; then
				echo "#SBATCH --mem=45GB" >> slurm_$flag".sh"
			else
				echo "#SBATCH --mem=13GB" >> slurm_$flag".sh"
			fi
			echo "#SBATCH --nodes=1" >> slurm_$flag".sh"
			if [ "$alg" == "bp_exp" ] || [ "$alg" == "bp_gamma" ]; then
				echo "#SBATCH --ntasks-per-node=32" >> slurm_$flag".sh"
			else
				echo "#SBATCH --ntasks-per-node=1" >> slurm_$flag".sh"
			fi
			echo "#SBATCH --output " $PRINTDIR"log_"$flag".out" >> slurm_$flag".sh"
			echo "#SBATCH -e " $PRINTDIR"log_"$flag".err" >> slurm_$flag".sh"

			echo submitting job $flag 
			echo "module load gsl/2.5" >> slurm_$flag".sh"
			echo "module load intel/python/3/2019.4.088/" >> slurm_$flag".sh"
			echo "cd /home/amuntoni/sibyl-team/loop_mtpp/" >> slurm_$flag".sh" 
			#start_time=$(date +%s)
			#echo "python3 " $script "--fn_rate" $fn_rate "-m" $seed "-p" $prob_th "-n" $nobs "-b " $beta " -i " $ti " --adp_frac " $adp_frac >> slurm_$flag".sh"	
			echo "python3 " $script "--fn_rate" $fn_rate "--tau " $tau " -m " $seed  "-n" $nobs "-b " $beta " -i " $ti " --adp_frac " $adp_frac >> "slurm_"$flag".sh"	

			sbatch "slurm_"$flag".sh"

			mv "slurm_"$flag".sh" "slurm_files/"

			sleep 2.0
			
		done
	done
done
