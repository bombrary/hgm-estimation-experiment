def v_phis_cache(y, mu, sig):
    match (y, mu, sig):
        case (0.01, 0.01, 1.0):
            v_phi00 = [ -2.26946961e-04
                      , -1.82199367e-04
                      , 3.10209292e-03
                      , -8.15620917e-05
                      , 1.12648231e-03
                      , -4.11481246e-04
                      , 2.93652370e-01]

            v_phi10 = [ 4.06999348e-02
                      , 1.17295408e-02
                      , -1.52413311e-04
                      , 2.32589161e-01
                      , 2.33243840e-01
                      , 5.24385018e-04
                      , 4.65850780e-03]

            v_phi20 = [ -0.00082662
                      , 0.00072744
                      , 0.01242631
                      , 0.00168018
                      , 0.00774273
                      , 0.21064475
                      , 0.47818713]

        case (0.01, -0.01, 1.0):
            v_phi00 = [ 0.000261477839380575
                      , -0.0003062405884435293
                      , 0.0030994657951310955
                      , -0.0023507566595248885
                      , 0.0011428520686196375
                      , -0.0004065966697219814
                      , 0.2936296768203743]

            v_phi10 = [ 0.04070562964164469
                      , 0.011734400137942374
                      , -9.054336968023131e-05
                      , 0.23260720350028094
                      , 0.23322755716679036
                      , 0.0002897414628382261
                      , -6.190448978976857e-06]

            v_phi20 = [ -0.001784279654581944
                      , 0.00023029676776697983
                      , 0.01243017094809673
                      , -0.0053790846417378635
                      , -0.0006828314421114534
                      , 0.2106351766855974
                      , 0.4781165297519957]

        case (-0.01, 0.01, 1.0):
             v_phi00 = [-0.000261477839380575
                       , 0.0003062405884435293
                       , 0.0030994657951310955
                       , 0.0023507566595248885
                       , -0.0011428520686196375
                       , -0.0004065966697219814
                       , 0.2936296768203743]

             v_phi10 = [0.04070562964164469
                       , 0.011734400137942374
                       , 9.054336968023131e-05
                       , 0.23260720350028094
                       , 0.23322755716679036
                       , -0.0002897414628382261
                       , 6.190448978976857e-06]

             v_phi20 = [0.001784279654581944
                       , -0.00023029676776697983
                       , 0.01243017094809673
                       , 0.0053790846417378635
                       , 0.0006828314421114534
                       , 0.2106351766855974
                       , 0.4781165297519957]

        case (-0.01, -0.01, 1.0):
            v_phi00 = [0.00022694696122371738
                      , 0.00018219936670504921
                      , 0.003102092915874266
                      , 8.15620917471449e-05
                      , -0.0011264823073509067
                      , -0.00041148124563239463
                      , 0.293652370491403]
            v_phi10 = [0.0406999347553165
                      , 0.01172954081562691
                      , 0.00015241331113857726
                      , 0.23258916114927583
                      , 0.23324384001825954
                      , -0.000524385017704361
                      , -0.004658507801699635]
            v_phi20 = [0.0008266162337955407
                      , -0.0007274431335080322
                      , 0.012426306483792615
                      , -0.0016801805892974198
                      , -0.007742734243143534
                      , 0.21064475434365293
                      , 0.47818712813829944]

        case t:
            raise NotImplementedError(f'Invalid variable: {t}')

    return v_phi00, v_phi10, v_phi20

