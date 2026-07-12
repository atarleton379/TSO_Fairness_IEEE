class InputHandler:
    def __init__(self):
        self.P_ch_max_bat = 150
        self.P_dis_max_bat = 150 
        self.E_cap_bat = 300 
        self.Effic_ch = 0.95 
        self.Effic_dis = 0.97 

        self.generators = {
            1: {
                "node": 1, "P_max": 152, "P_min": 0, "R_plus": 40, "R_minus": 40,
                "Ramp_up": 120, "Ramp_down": 120, "T_up": 8, "T_down": 4,
                "C": 13.32, "C_up": 15, "C_down": 14, "C_plus": 15, "C_minus": 11,
                "C_su": 1430.4, "P_ini": 76, "U_ini": 1, "T_ini": 22
            },
            2: {
                "node": 2, "P_max": 152, "P_min": 0, "R_plus": 40, "R_minus": 40,
                "Ramp_up": 120, "Ramp_down": 120, "T_up": 8, "T_down": 4,
                "C": 13.32, "C_up": 15, "C_down": 14, "C_plus": 15, "C_minus": 11,
                "C_su": 1430.4, "P_ini": 76, "U_ini": 1, "T_ini": 22
            },
            3: {
                "node": 7, "P_max": 350, "P_min": 0, "R_plus": 70, "R_minus": 70,
                "Ramp_up": 350, "Ramp_down": 350, "T_up": 8, "T_down": 8,
                "C": 20.7, "C_up": 10, "C_down": 9, "C_plus": 24, "C_minus": 16,
                "C_su": 1725, "P_ini": 0, "U_ini": 0, "T_ini": -2
            },
            4: {
                "node": 13, "P_max": 591, "P_min": 0, "R_plus": 180, "R_minus": 180,
                "Ramp_up": 240, "Ramp_down": 240, "T_up": 12, "T_down": 10,
                "C": 20.93, "C_up": 8, "C_down": 7, "C_plus": 25, "C_minus": 17,
                "C_su": 3056.7, "P_ini": 0, "U_ini": 0, "T_ini": -1
            },
            5: {
                "node": 15, "P_max": 60, "P_min": 0, "R_plus": 60, "R_minus": 60,
                "Ramp_up": 60, "Ramp_down": 60, "T_up": 4, "T_down": 2,
                "C": 26.11, "C_up": 7, "C_down": 5, "C_plus": 28, "C_minus": 23,
                "C_su": 437, "P_ini": 0, "U_ini": 0, "T_ini": -1
            },
            6: {
                "node": 15, "P_max": 155, "P_min": 0, "R_plus": 30, "R_minus": 30,
                "Ramp_up": 155, "Ramp_down": 155, "T_up": 8, "T_down": 8,
                "C": 10.52, "C_up": 16, "C_down": 14, "C_plus": 16, "C_minus": 7,
                "C_su": 312, "P_ini": 0, "U_ini": 0, "T_ini": -2
            },
            7: {
                "node": 16, "P_max": 155, "P_min": 0, "R_plus": 30, "R_minus": 30,
                "Ramp_up": 155, "Ramp_down": 155, "T_up": 8, "T_down": 8,
                "C": 10.52, "C_up": 16, "C_down": 14, "C_plus": 16, "C_minus": 7,
                "C_su": 312, "P_ini": 124, "U_ini": 1, "T_ini": 10
            },
            8: {
                "node": 18, "P_max": 400, "P_min": 0, "R_plus": 0, "R_minus": 0,
                "Ramp_up": 280, "Ramp_down": 280, "T_up": 1, "T_down": 1,
                "C": 6.02, "C_up": 0, "C_down": 0, "C_plus": 0, "C_minus": 0,
                "C_su": 0, "P_ini": 240, "U_ini": 1, "T_ini": 50
            },
            9: {
                "node": 21, "P_max": 400, "P_min": 0, "R_plus": 0, "R_minus": 0,
                "Ramp_up": 280, "Ramp_down": 280, "T_up": 1, "T_down": 1,
                "C": 5.47, "C_up": 0, "C_down": 0, "C_plus": 0, "C_minus": 0,
                "C_su": 0, "P_ini": 240, "U_ini": 1, "T_ini": 16
            },
            10: {
                "node": 22, "P_max": 300, "P_min": 0, "R_plus": 0, "R_minus": 0,
                "Ramp_up": 300, "Ramp_down": 300, "T_up": 0, "T_down": 0,
                "C": 0, "C_up": 0, "C_down": 0, "C_plus": 0, "C_minus": 0,
                "C_su": 0, "P_ini": 240, "U_ini": 1, "T_ini": 24
            },
            11: {
                "node": 23, "P_max": 310, "P_min": 0, "R_plus": 60, "R_minus": 60,
                "Ramp_up": 180, "Ramp_down": 180, "T_up": 8, "T_down": 8,
                "C": 10.52, "C_up": 17, "C_down": 16, "C_plus": 14, "C_minus": 8,
                "C_su": 624, "P_ini": 248, "U_ini": 1, "T_ini": 10
            },
            12: {
                "node": 23, "P_max": 350, "P_min": 0, "R_plus": 40, "R_minus": 40,
                "Ramp_up": 240, "Ramp_down": 240, "T_up": 8, "T_down": 8,
                "C": 10.89, "C_up": 16, "C_down": 14, "C_plus": 16, "C_minus": 8,
                "C_su": 2298, "P_ini": 280, "U_ini": 1, "T_ini": 50
            }
        }

        self.wind_farms = {
            1: {"node": 3, "P_max": 200},
            2: {"node": 5, "P_max": 200},
            3: {"node": 7, "P_max": 200},
            4: {"node": 16, "P_max": 200},
            5: {"node": 21, "P_max": 200},
            6: {"node": 23, "P_max": 200},
            }

        # from https://zenodo.org/records/3253876#.XSiVOEdS8l0, 2015-09-25, Denmark
        self.CF_wind = {
            1: 0.258,
            2: 0.252,
            3: 0.246,
            4: 0.249,
            5: 0.25,
            6: 0.251,
            7: 0.251,
            8: 0.281,
            9: 0.3,
            10: 0.344,
            11: 0.378,
            12: 0.399,
            13: 0.424,
            14: 0.382,
            15: 0.321,
            16: 0.209,
            17: 0.145,
            18: 0.128,
            19: 0.108,
            20: 0.117,
            21: 0.128,
            22: 0.129,
            23: 0.133,
            24: 0.137}

        self.demands = {
            "system_load": {
                1: 1775.835, 2: 1669.815, 3: 1590.3, 4: 1563.795,
                5: 1563.795, 6: 1590.3, 7: 1961.37, 8: 2279.43,
                9: 2517.975, 10: 2544.48, 11: 2544.48, 12: 2517.975,
                13: 2517.975, 14: 2517.975, 15: 2464.965, 16: 2464.965,
                17: 2623.995, 18: 2650.5, 19: 2650.5, 20: 2544.48,
                21: 2411.955, 22: 2199.915, 23: 1934.865, 24: 1669.815
            },
            "load_distribution": {
                1: 0.038, 2: 0.034, 3: 0.063, 4: 0.026, 5: 0.025,
                6: 0.048, 7: 0.044, 8: 0.060, 9: 0.061,
                10: 0.068, 11: 0.093, 12: 0.068, 13: 0.111,
                14: 0.035, 15: 0.117, 16: 0.064, 17: 0.045
            },
            "bidding_prices": {
                1: 45, 2: 38, 3: 30, 4: 25, 5: 20, # 20 is the lowest, most of them are above the maximum bidding price, so almost all of them will be fulfilled unless network constraints take effect
                6: 45, 7: 38, 8: 38, 9: 30,
                10: 30, 11: 25, 12: 20, 13: 45,
                14: 38, 15: 38, 16: 30, 17: 25
            },
            "load_location": {
                1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 
                7: 7, 8: 8, 9: 9, 10: 10, 11: 13, 12: 14, 
                13: 15, 14: 16, 15: 18, 16: 19, 17: 20
            }
        }

        self.lines = {                                              # adjusted reactance according to assumptions in the problem statement
            1: {"from": 1, "to": 2, "x": 0.002, "capacity": 175},
            2: {"from": 1, "to": 3, "x": 0.002, "capacity": 175},
            3: {"from": 1, "to": 5, "x": 0.002, "capacity": 350},
            4: {"from": 2, "to": 4, "x": 0.002, "capacity": 175},
            5: {"from": 2, "to": 6, "x": 0.002, "capacity": 175},
            6: {"from": 3, "to": 9, "x": 0.002, "capacity": 175},
            7: {"from": 3, "to": 24, "x": 0.002, "capacity": 400},
            8: {"from": 4, "to": 9, "x": 0.002, "capacity": 175},
            9: {"from": 5, "to": 10, "x": 0.002, "capacity": 350},
            10: {"from": 6, "to": 10, "x": 0.002, "capacity": 175},
            11: {"from": 7, "to": 8, "x": 0.002, "capacity": 350},
            12: {"from": 8, "to": 9, "x": 0.002, "capacity": 175},
            13: {"from": 8, "to": 10, "x": 0.002, "capacity": 175},
            14: {"from": 9, "to": 11, "x": 0.002, "capacity": 400},
            15: {"from": 9, "to": 12, "x": 0.002, "capacity": 400},
            16: {"from": 10, "to": 11, "x": 0.002, "capacity": 400},
            17: {"from": 10, "to": 12, "x": 0.002, "capacity": 400},
            18: {"from": 11, "to": 13, "x": 0.002, "capacity": 500},
            19: {"from": 11, "to": 14, "x": 0.002, "capacity": 500},
            20: {"from": 12, "to": 13, "x": 0.002, "capacity": 500},
            21: {"from": 12, "to": 23, "x": 0.002, "capacity": 500},
            22: {"from": 13, "to": 23, "x": 0.002, "capacity": 250}, # Previously 500 MW, adjusted according to section 3 in paper
            23: {"from": 14, "to": 16, "x": 0.002, "capacity": 250}, # Previously 500 MW, adjusted according to section 3 in paper
            24: {"from": 15, "to": 16, "x": 0.002, "capacity": 500},
            25: {"from": 15, "to": 21, "x": 0.002, "capacity": 400}, # Previously 1000MW, adjusted according to section 3 in paper
            26: {"from": 15, "to": 24, "x": 0.002, "capacity": 500},
            27: {"from": 16, "to": 17, "x": 0.002, "capacity": 500},
            28: {"from": 16, "to": 19, "x": 0.002, "capacity": 500},
            29: {"from": 17, "to": 18, "x": 0.002, "capacity": 500},
            30: {"from": 17, "to": 22, "x": 0.002, "capacity": 500},
            31: {"from": 18, "to": 21, "x": 0.002, "capacity": 1000},
            32: {"from": 19, "to": 20, "x": 0.002, "capacity": 1000},
            33: {"from": 20, "to": 23, "x": 0.002, "capacity": 1000},
            34: {"from": 21, "to": 22, "x": 0.002, "capacity": 500}
        }

        self.sys_base = 100 # MVA

        self.nodes = [
            1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24
        ]
        # self.zones = {
        #     1: {18,21,22,17,23,16,19,20},
        #     2: {15,14,13,24,11,12},
        #     3: {3,9,10,6,4,5,1,2,7,8}  
        # }

        self.zones = {
            1: {21,22,23,19,20,13},
            2: {18,17,16,15,14,24,11,3},
            3: {12,9,10,4,5,8,6,1,2,7}  
        }

        # self.zones = {
        #     1: [1],
        #     2: [2],
        #     3: [3],
        #     4: [4],
        #     5: [5],
        #     6: [6],
        #     7: [7],
        #     8: [8],
        #     9: [9],
        #     10: [10],
        #     11: [11],
        #     12: [12],
        #     13: [13],
        #     14: [14],
        #     15: [15],
        #     16: [16],
        #     17: [17],
        #     18: [18],
        #     19: [19],
        #     20: [20],
        #     21: [21],
        #     22: [22],
        #     23: [23],
        #     24: [24],
        # }