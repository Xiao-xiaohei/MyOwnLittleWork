#!/usr/bin

echo "----Start----"

expect -c "
	set timeout 120

	spawn scp -r yzy@162.105.92.95:/mnt/hd6t/yzy/project/RSHNet.0.pth_psm_not_normalization_T_bins_2.npy /Users/yuanzeyu/Desktop/new/Lab_work/code

	expect {
		\"(yes/no)?\" {send \"yes\n\"; expect \"*assword:\" { send \"142766364\n\";}}
	\"*assword:\" {
	send \"142766364\n\";
	}
}
expect \"100%\"
expect eof
"

expect -c "
	set timeout 120

	spawn scp -r yzy@162.105.92.95:/mnt/hd6t/yzy/project/RSHNet.0.pth_psm_not_normalization_T_bins_3.npy /Users/yuanzeyu/Desktop/new/Lab_work/code

	expect {
		\"(yes/no)?\" {send \"yes\n\"; expect \"*assword:\" { send \"142766364\n\";}}
	\"*assword:\" {
	send \"142766364\n\";
	}
}
expect \"100%\"
expect eof
"

expect -c "
	set timeout 120

	spawn scp -r yzy@162.105.92.95:/mnt/hd6t/yzy/project/RSHNet.0.pth_psm_not_normalization_T_bins_4.npy /Users/yuanzeyu/Desktop/new/Lab_work/code

	expect {
		\"(yes/no)?\" {send \"yes\n\"; expect \"*assword:\" { send \"142766364\n\";}}
	\"*assword:\" {
	send \"142766364\n\";
	}
}
expect \"100%\"
expect eof
"

echo "----Done----"
