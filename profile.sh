# ncu -o /szymon.ozog/profile$BS -f --kernel-id ::regex:'^(?!.*elementwise).*': --set full python run_moe.py $BS
ncu -o /szymon.ozog/profile$BS -f --kernel-id ::regex:'fused_moe*': --set full python run_moe.py $BS

