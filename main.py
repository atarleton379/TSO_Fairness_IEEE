from input import InputHandler
from model1 import Model1
from model2 import Model2
from model3_nodal import Model3Nodal
from model3_zonal import Model3Zonal
from model5 import Model5
from model6 import Model6
from model6_bonus import Model6Bonus
from model6_bonus_split_bid import Model6Bonus_split_bid
from results_saver import save_results
import numpy as np
HOUR = 12
HOURS = 24
sensitivity_factor = 1
inp_hndl = InputHandler()
model1 = Model1(HOUR, inp_hndl)
model2 = Model2(HOURS, inp_hndl)
model3_nodal = Model3Nodal(HOUR, inp_hndl, sensitivity_factor)
model3_zonal = Model3Zonal(HOUR, inp_hndl, sensitivity_factor)
model5 = Model5(HOUR, inp_hndl)
model6 = Model6(HOUR, inp_hndl)
model6_bonus = Model6Bonus(HOUR, inp_hndl)




###### Sensitivity analysis iterators########
# sensitivity_range = np.linspace(0.5, 1.5, 11)
# model3_sens_analysis = {}
# model4_sens_analysis = {}
# sens_results = {}
# for sens in sensitivity_range:
#     model3_sens_analysis[sens] = Model3(HOUR, inp_hndl, sensitivity_factor=sens)
#     sens_results[f"model3_sens_{sens}"] = model3_sens_analysis[sens].out_dict
#     model4_sens_analysis[sens] = Model4(HOUR, inp_hndl, sensitivity_factor=sens)
#     sens_results[f"model4_sens_{sens}"] = model4_sens_analysis[sens].out_dict
# save_results(sens_results, base_folder="code/results/sensitivity", run_name="sensitivity")

###### Terminal Output functions, can delete if in the way and definitely before submitting###########################################
# #Model 1 outputs
# print('Model 1 market clearing price')
# print(model1.out_dict['market_clearing_price'])

# #Model 3 outputs
# print('Model 3 Nodal Prices')
# for n in model3.Nodes:
#     print(f"node {n} : {-model3.out_dict[f"node_{n}_price"]}")
# print('Model 3 line flows')
# for i in model3.Lines:
#     print(f'line {i} : {model3.out_dict[f"line_{i}_flow"]}')

# #Model 4 Outputs
# print('Model 4 zonal Prices')
# for n in model4.Zones:
#     print(f"node {n} : {-model4.out_dict[f"zone_{n}_price"]}")
# print('Model 4 zone flows')
# for pair in model4.zone_pairs:
#     print(f"zone {pair} : {model4.out_dict[f"zone_{pair[0]}_{pair[1]}_flow"]}")

save_results({
    "model1": model1.out_dict,
    "model2": model2.out_dict,
    "model3_nodal": model3_nodal.out_dict,
    "model3_zonal": model3_zonal.out_dict,
    "model5": model5.out_dict,
    "model6_bonus": model6_bonus.out_dict
}, base_folder="code/results")

# model5.out_fig[0][0].savefig(f"{results_folder}/model5_balancing_merit_order.png", dpi=150)
# model5.out_fig[1].savefig(f"{results_folder}/model5_revenue_breakdown.png", dpi=150)