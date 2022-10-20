import numpy as np
cimport numpy as cnp
cimport cython      

ctypedef cnp.float64_t DTYPE_t

cpdef inline cnp.ndarray[DTYPE_t, ndim=3] phi0(DTYPE_t[:] z):
    cdef DTYPE_t y = z[0]
    cdef DTYPE_t mu = z[1]
    cdef DTYPE_t sig = z[2]
    return np.array([[[(-mu*y-2/5*mu**2+2/5*sig)/(mu),-24/25*mu**3+(24/25*sig+2)*mu,(-16/25*sig-1)*mu**2+16/25*sig**2+3*sig+25/8,(256/5*mu**4+(-512/5*sig-480)*mu**2+256/5*sig**2+160*sig)/((256*sig**2+800*sig+625)*mu),(((-512/5*sig-160)*mu**3+(512/5*sig**2+160*sig)*mu)*y-1024/25*mu**6+(-2048/25*sig-256)*mu**4+(3072/25*sig**2+640*sig+600)*mu**2-128*sig**2-200*sig)/((256*sig**2+800*sig+625)*mu),(((-32/5*sig-10)*mu**2+32/5*sig**2+10*sig)*y-384/25*mu**5-16*mu**3+(384/25*sig**2+64*sig+75)*mu)/((16*sig+25)*mu),((-256/5*mu**4-160*mu**2+256/5*sig**2+160*sig)*y+128*mu**3-128*sig*mu)/((256*sig**2+800*sig+625)*mu)],[(-24*mu**2+24*sig+50)/((16*sig+25)*mu),-y,0,(1024*mu**4+3200*mu**2+3072*sig**2+16000*sig+20000)/((4096*sig**3+19200*sig**2+30000*sig+15625)*mu),(160*mu**2-480*sig-1000)/((256*sig**2+800*sig+625)*mu),((-24*mu**2+24*sig+50)*y+40*mu)/((16*sig+25)*mu),((1024*mu**4+3200*mu**2+3072*sig**2+16000*sig+20000)*y-2560*mu**3+(-2560*sig-8000)*mu)/((4096*sig**3+19200*sig**2+30000*sig+15625)*mu)],[(256*mu**2-768*sig-1600)/(256*sig**2+800*sig+625),(20)/(16*sig+25),-y,(-16384*mu**4+32768*sig*mu**2-49152*sig**2-204800*sig-240000)/(65536*sig**4+409600*sig**3+960000*sig**2+1000000*sig+390625),(-2560*mu**2+7680*sig+16000)/(4096*sig**3+19200*sig**2+30000*sig+15625),((256*mu**2-768*sig-1600)*y-320*mu)/(256*sig**2+800*sig+625),((-16384*mu**4+32768*sig*mu**2-49152*sig**2-204800*sig-240000)*y+40960*mu**3-40960*sig*mu)/(65536*sig**4+409600*sig**3+960000*sig**2+1000000*sig+390625)],[(-4/5*sig**2-5/2*sig-125/64)/(mu),(-48/25*sig**2-6*sig-75/16)*mu,-32/25*sig**3-6*sig**2-75/8*sig-625/128,(-mu*y+2/5*mu**2-2/5*sig-5/4)/(mu),((-4/5*sig-5/4)*mu*y-8/25*mu**4+(-24/25*sig-2)*mu**2+sig+25/16)/(mu),((-4/5*sig**2-5/2*sig-125/64)*y+(-48/25*sig-3)*mu**3+(-48/25*sig**2-7*sig-25/4)*mu)/(mu),((-2/5*mu**2-2/5*sig-5/4)*y)/(mu)],[(-sig-25/16)/(mu),0,0,(-8*mu**2-8*sig-25)/((16*sig+25)*mu),(-mu*y+5/4)/(mu),((-sig-25/16)*y)/(mu),((-8*mu**2-8*sig-25)*y+20*mu)/((16*sig+25)*mu)],[1,0,0,0,0,0,0],[0,0,0,1,0,0,0]],[[(-24*mu**2+24*sig+50)/((16*sig+25)*mu),-y,0,(1024*mu**4+3200*mu**2+3072*sig**2+16000*sig+20000)/((4096*sig**3+19200*sig**2+30000*sig+15625)*mu),(160*mu**2-480*sig-1000)/((256*sig**2+800*sig+625)*mu),((-24*mu**2+24*sig+50)*y+40*mu)/((16*sig+25)*mu),((1024*mu**4+3200*mu**2+3072*sig**2+16000*sig+20000)*y-2560*mu**3+(-2560*sig-8000)*mu)/((4096*sig**3+19200*sig**2+30000*sig+15625)*mu)],[0,0,2,0,0,0,0],[0,(-768*mu**2-768*sig-1600)/(256*sig**2+800*sig+625),(-64*mu)/(16*sig+25),(-128000)/(65536*sig**4+409600*sig**3+960000*sig**2+1000000*sig+390625),((-81920*sig-128000)*mu*y-16384*mu**4+(-98304*sig-204800)*mu**2-49152*sig**2-102400*sig-80000)/(65536*sig**4+409600*sig**3+960000*sig**2+1000000*sig+390625),((-5120*sig-8000)*y-8192*mu**3+(-24576*sig-51200)*mu)/(4096*sig**3+19200*sig**2+30000*sig+15625),((-40960*mu**2-40960*sig-128000)*y+102400*mu)/(65536*sig**4+409600*sig**3+960000*sig**2+1000000*sig+390625)],[(-sig-25/16)/(mu),0,0,(-8*mu**2-8*sig-25)/((16*sig+25)*mu),(-mu*y+5/4)/(mu),((-sig-25/16)*y)/(mu),((-8*mu**2-8*sig-25)*y+20*mu)/((16*sig+25)*mu)],[0,0,0,0,0,2,0],[0,1,0,0,0,0,0],[0,0,0,0,1,0,0]],[[(256*mu**2-768*sig-1600)/(256*sig**2+800*sig+625),(20)/(16*sig+25),-y,(-16384*mu**4+32768*sig*mu**2-49152*sig**2-204800*sig-240000)/(65536*sig**4+409600*sig**3+960000*sig**2+1000000*sig+390625),(-2560*mu**2+7680*sig+16000)/(4096*sig**3+19200*sig**2+30000*sig+15625),((256*mu**2-768*sig-1600)*y-320*mu)/(256*sig**2+800*sig+625),((-16384*mu**4+32768*sig*mu**2-49152*sig**2-204800*sig-240000)*y+40960*mu**3-40960*sig*mu)/(65536*sig**4+409600*sig**3+960000*sig**2+1000000*sig+390625)],[0,(-768*mu**2-768*sig-1600)/(256*sig**2+800*sig+625),(-64*mu)/(16*sig+25),(-128000)/(65536*sig**4+409600*sig**3+960000*sig**2+1000000*sig+390625),((-81920*sig-128000)*mu*y-16384*mu**4+(-98304*sig-204800)*mu**2-49152*sig**2-102400*sig-80000)/(65536*sig**4+409600*sig**3+960000*sig**2+1000000*sig+390625),((-5120*sig-8000)*y-8192*mu**3+(-24576*sig-51200)*mu)/(4096*sig**3+19200*sig**2+30000*sig+15625),((-40960*mu**2-40960*sig-128000)*y+102400*mu)/(65536*sig**4+409600*sig**3+960000*sig**2+1000000*sig+390625)],[(4000)/((4096*sig**3+19200*sig**2+30000*sig+15625)*mu),((-2560*sig-4000)*y+20480*mu**3+6400*mu)/(4096*sig**3+19200*sig**2+30000*sig+15625),(1280*mu**2-1280*sig-2400)/(256*sig**2+800*sig+625),(4608000*mu**2+512000*sig+1600000)/((1048576*sig**5+8192000*sig**4+25600000*sig**3+40000000*sig**2+31250000*sig+9765625)*mu),(((2293760*sig+3584000)*mu**3+(-983040*sig**2-2560000*sig-1600000)*mu)*y+524288*mu**6+(2621440*sig+5734400)*mu**4+(-1638400*sig-1280000)*mu**2-1280000*sig-2000000)/((1048576*sig**5+8192000*sig**4+25600000*sig**3+40000000*sig**2+31250000*sig+9765625)*mu),(((81920*sig+128000)*mu**2+64000*sig+100000)*y+245760*mu**5+(491520*sig+1126400)*mu**3+(-245760*sig**2-819200*sig-720000)*mu)/((65536*sig**4+409600*sig**3+960000*sig**2+1000000*sig+390625)*mu),((1310720*mu**4+(655360*sig+3584000)*mu**2+512000*sig+1600000)*y-3276800*mu**3+819200*sig*mu)/((1048576*sig**5+8192000*sig**4+25600000*sig**3+40000000*sig**2+31250000*sig+9765625)*mu)],[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,0,0,1,0]]])


cpdef inline cnp.ndarray[DTYPE_t, ndim=3] phi1(DTYPE_t[:] z):
    cdef DTYPE_t y = z[0]
    cdef DTYPE_t mu = z[1]
    cdef DTYPE_t sig = z[2]
    return np.array([[[((-16*sig-25)*mu*y**2+((-32/5*sig-20)*mu**2+32/5*sig**2+10*sig)*y-4*mu**3+(20*sig+25)*mu)/((16*sig+25)*mu*y+10*mu**2),(((-384/25*sig-24)*mu**3+(384/25*sig**2+56*sig+50)*mu)*y-48/5*mu**4+(144/5*sig+50)*mu**2)/((16*sig+25)*y+10*mu),(((-256/25*sig**2-32*sig-25)*mu**2+256/25*sig**3+64*sig**2+125*sig+625/8)*y+(-32/5*sig-10)*mu**3+(96/5*sig**2+70*sig+125/2)*mu)/((16*sig+25)*y+10*mu),(((4096/5*sig+1280)*mu**4+(-8192/5*sig**2-10240*sig-12000)*mu**2+4096/5*sig**3+3840*sig**2+4000*sig)*y+512*mu**5+(-3072*sig-8000)*mu**3+(2560*sig**2+11200*sig+10000)*mu)/((4096*sig**3+19200*sig**2+30000*sig+15625)*mu*y+(2560*sig**2+8000*sig+6250)*mu**2),(((-8192/5*sig**2-5120*sig-4000)*mu**3+(8192/5*sig**3+5120*sig**2+4000*sig)*mu)*y**2+((-16384/25*sig-1024)*mu**6+(-3072*sig-4800)*mu**4+(16384/25*sig**3+10240*sig**2+24000*sig+15000)*mu**2-2048*sig**3-6400*sig**2-5000*sig)*y-2048/5*mu**7+4096/5*sig*mu**5+(6144/5*sig**2+8960*sig+10000)*mu**3+(-6400*sig**2-18000*sig-12500)*mu)/((4096*sig**3+19200*sig**2+30000*sig+15625)*mu*y+(2560*sig**2+8000*sig+6250)*mu**2),(((-512/5*sig**2-320*sig-250)*mu**2+512/5*sig**3+320*sig**2+250*sig)*y**2+((-6144/25*sig-384)*mu**5+(2048/25*sig**2-64*sig-300)*mu**3+(4096/25*sig**3+1216*sig**2+2700*sig+1875)*mu)*y-768/5*mu**6+(1792/5*sig+400)*mu**4+(1536/5*sig**2+1360*sig+1500)*mu**2)/((256*sig**2+800*sig+625)*mu*y+(160*sig+250)*mu**2),(((-4096/5*sig-1280)*mu**4+(-2560*sig-4000)*mu**2+4096/5*sig**3+3840*sig**2+4000*sig)*y**2+((16384/25*sig+512)*mu**5+(4096*sig+4800)*mu**3+(-16384/25*sig**3-4608*sig**2-4800*sig)*mu)*y+2048/5*mu**6+(-4096/5*sig+1280)*mu**4+(-6144/5*sig**2-8960*sig-8000)*mu**2)/((4096*sig**3+19200*sig**2+30000*sig+15625)*mu*y+(2560*sig**2+8000*sig+6250)*mu**2)],[(-24*mu**2+24*sig+50)/((16*sig+25)*mu),-y,0,(1024*mu**4+3200*mu**2+3072*sig**2+16000*sig+20000)/((4096*sig**3+19200*sig**2+30000*sig+15625)*mu),(160*mu**2-480*sig-1000)/((256*sig**2+800*sig+625)*mu),((-24*mu**2+24*sig+50)*y+40*mu)/((16*sig+25)*mu),((1024*mu**4+3200*mu**2+3072*sig**2+16000*sig+20000)*y-2560*mu**3+(-2560*sig-8000)*mu)/((4096*sig**3+19200*sig**2+30000*sig+15625)*mu)],[(256*mu**2-768*sig-1600)/(256*sig**2+800*sig+625),(20)/(16*sig+25),-y,(-16384*mu**4+32768*sig*mu**2-49152*sig**2-204800*sig-240000)/(65536*sig**4+409600*sig**3+960000*sig**2+1000000*sig+390625),(-2560*mu**2+7680*sig+16000)/(4096*sig**3+19200*sig**2+30000*sig+15625),((256*mu**2-768*sig-1600)*y-320*mu)/(256*sig**2+800*sig+625),((-16384*mu**4+32768*sig*mu**2-49152*sig**2-204800*sig-240000)*y+40960*mu**3-40960*sig*mu)/(65536*sig**4+409600*sig**3+960000*sig**2+1000000*sig+390625)],[(-4/5*sig**2-5/2*sig-125/64)/(mu),(-48/25*sig**2-6*sig-75/16)*mu,-32/25*sig**3-6*sig**2-75/8*sig-625/128,(-mu*y+2/5*mu**2-2/5*sig-5/4)/(mu),((-4/5*sig-5/4)*mu*y-8/25*mu**4+(-8/25*sig-1)*mu**2+sig+25/16)/(mu),((-4/5*sig**2-5/2*sig-125/64)*y+(-48/25*sig-3)*mu**3+(-32/25*sig**2-5*sig-75/16)*mu)/(mu),((-2/5*mu**2-2/5*sig-5/4)*y+8/25*mu**3+(8/25*sig+1)*mu)/(mu)],[(-sig-25/16)/(mu),0,0,(-8*mu**2-8*sig-25)/((16*sig+25)*mu),(-mu*y+5/4)/(mu),((-sig-25/16)*y)/(mu),((-8*mu**2-8*sig-25)*y+20*mu)/((16*sig+25)*mu)],[1,0,0,0,0,0,0],[0,0,0,1,0,0,0]],[[(-24*mu**2+24*sig+50)/((16*sig+25)*mu),-y,0,(1024*mu**4+3200*mu**2+3072*sig**2+16000*sig+20000)/((4096*sig**3+19200*sig**2+30000*sig+15625)*mu),(160*mu**2-480*sig-1000)/((256*sig**2+800*sig+625)*mu),((-24*mu**2+24*sig+50)*y+40*mu)/((16*sig+25)*mu),((1024*mu**4+3200*mu**2+3072*sig**2+16000*sig+20000)*y-2560*mu**3+(-2560*sig-8000)*mu)/((4096*sig**3+19200*sig**2+30000*sig+15625)*mu)],[0,0,2,0,0,0,0],[(200)/((256*sig**2+800*sig+625)*mu*y+(160*sig+250)*mu**2),(((-12288*sig-19200)*mu**2-10240*sig**2-38400*sig-35000)*y-7680*mu**3+(-2560*sig-8000)*mu)/((4096*sig**3+19200*sig**2+30000*sig+15625)*y+(2560*sig**2+8000*sig+6250)*mu),((-1024*sig-1600)*mu*y-640*mu**2+160*sig+250)/((256*sig**2+800*sig+625)*y+(160*sig+250)*mu),((-2048000*sig-3200000)*mu*y+(-409600*sig-1920000)*mu**2+409600*sig**2+1920000*sig+2000000)/((1048576*sig**5+8192000*sig**4+25600000*sig**3+40000000*sig**2+31250000*sig+9765625)*mu*y+(655360*sig**4+4096000*sig**3+9600000*sig**2+10000000*sig+3906250)*mu**2),((-1310720*sig**2-4096000*sig-3200000)*mu**2*y**2+((-262144*sig-409600)*mu**5+(-786432*sig**2-4096000*sig-4480000)*mu**3+(2048000*sig**2+5760000*sig+4000000)*mu)*y-163840*mu**6+(-327680*sig-1024000)*mu**4+(163840*sig**2+2048000*sig+2400000)*mu**2-1024000*sig**2-3200000*sig-2500000)/((1048576*sig**5+8192000*sig**4+25600000*sig**3+40000000*sig**2+31250000*sig+9765625)*mu*y+(655360*sig**4+4096000*sig**3+9600000*sig**2+10000000*sig+3906250)*mu**2),((-81920*sig**2-256000*sig-200000)*mu*y**2+((-131072*sig-204800)*mu**4+(-294912*sig**2-1177600*sig-1120000)*mu**2+51200*sig**2+160000*sig+125000)*y-81920*mu**5+(-122880*sig-320000)*mu**3+(40960*sig**2+192000*sig+200000)*mu)/((65536*sig**4+409600*sig**3+960000*sig**2+1000000*sig+390625)*mu*y+(40960*sig**3+192000*sig**2+300000*sig+156250)*mu**2),(((-655360*sig-1024000)*mu**3+(-655360*sig**2-3072000*sig-3200000)*mu)*y**2+(262144*sig*mu**4+(786432*sig**2+4505600*sig+4480000)*mu**2+409600*sig**2+1920000*sig+2000000)*y+163840*mu**5+(327680*sig+2048000)*mu**3+(-163840*sig**2-1024000*sig-800000)*mu)/((1048576*sig**5+8192000*sig**4+25600000*sig**3+40000000*sig**2+31250000*sig+9765625)*mu*y+(655360*sig**4+4096000*sig**3+9600000*sig**2+10000000*sig+3906250)*mu**2)],[(-sig-25/16)/(mu),0,0,(-8*mu**2-8*sig-25)/((16*sig+25)*mu),(-mu*y+5/4)/(mu),((-sig-25/16)*y)/(mu),((-8*mu**2-8*sig-25)*y+20*mu)/((16*sig+25)*mu)],[0,0,0,0,0,2,0],[0,1,0,0,0,0,0],[0,0,0,0,1,0,0]],[[(256*mu**2-768*sig-1600)/(256*sig**2+800*sig+625),(20)/(16*sig+25),-y,(-16384*mu**4+32768*sig*mu**2-49152*sig**2-204800*sig-240000)/(65536*sig**4+409600*sig**3+960000*sig**2+1000000*sig+390625),(-2560*mu**2+7680*sig+16000)/(4096*sig**3+19200*sig**2+30000*sig+15625),((256*mu**2-768*sig-1600)*y-320*mu)/(256*sig**2+800*sig+625),((-16384*mu**4+32768*sig*mu**2-49152*sig**2-204800*sig-240000)*y+40960*mu**3-40960*sig*mu)/(65536*sig**4+409600*sig**3+960000*sig**2+1000000*sig+390625)],[(200)/((256*sig**2+800*sig+625)*mu*y+(160*sig+250)*mu**2),(((-12288*sig-19200)*mu**2-10240*sig**2-38400*sig-35000)*y-7680*mu**3+(-2560*sig-8000)*mu)/((4096*sig**3+19200*sig**2+30000*sig+15625)*y+(2560*sig**2+8000*sig+6250)*mu),((-1024*sig-1600)*mu*y-640*mu**2+160*sig+250)/((256*sig**2+800*sig+625)*y+(160*sig+250)*mu),((-2048000*sig-3200000)*mu*y+(-409600*sig-1920000)*mu**2+409600*sig**2+1920000*sig+2000000)/((1048576*sig**5+8192000*sig**4+25600000*sig**3+40000000*sig**2+31250000*sig+9765625)*mu*y+(655360*sig**4+4096000*sig**3+9600000*sig**2+10000000*sig+3906250)*mu**2),((-1310720*sig**2-4096000*sig-3200000)*mu**2*y**2+((-262144*sig-409600)*mu**5+(-786432*sig**2-4096000*sig-4480000)*mu**3+(2048000*sig**2+5760000*sig+4000000)*mu)*y-163840*mu**6+(-327680*sig-1024000)*mu**4+(163840*sig**2+2048000*sig+2400000)*mu**2-1024000*sig**2-3200000*sig-2500000)/((1048576*sig**5+8192000*sig**4+25600000*sig**3+40000000*sig**2+31250000*sig+9765625)*mu*y+(655360*sig**4+4096000*sig**3+9600000*sig**2+10000000*sig+3906250)*mu**2),((-81920*sig**2-256000*sig-200000)*mu*y**2+((-131072*sig-204800)*mu**4+(-294912*sig**2-1177600*sig-1120000)*mu**2+51200*sig**2+160000*sig+125000)*y-81920*mu**5+(-122880*sig-320000)*mu**3+(40960*sig**2+192000*sig+200000)*mu)/((65536*sig**4+409600*sig**3+960000*sig**2+1000000*sig+390625)*mu*y+(40960*sig**3+192000*sig**2+300000*sig+156250)*mu**2),(((-655360*sig-1024000)*mu**3+(-655360*sig**2-3072000*sig-3200000)*mu)*y**2+(262144*sig*mu**4+(786432*sig**2+4505600*sig+4480000)*mu**2+409600*sig**2+1920000*sig+2000000)*y+163840*mu**5+(327680*sig+2048000)*mu**3+(-163840*sig**2-1024000*sig-800000)*mu)/((1048576*sig**5+8192000*sig**4+25600000*sig**3+40000000*sig**2+31250000*sig+9765625)*mu*y+(655360*sig**4+4096000*sig**3+9600000*sig**2+10000000*sig+3906250)*mu**2)],[((64000*sig+100000)*y+(-128000*sig-160000)*mu)/((65536*sig**4+409600*sig**3+960000*sig**2+1000000*sig+390625)*mu*y+(40960*sig**3+192000*sig**2+300000*sig+156250)*mu**2),((-40960*sig**2-128000*sig-100000)*y**2+((327680*sig+512000)*mu**3+(-16384*sig**2+25600*sig+80000)*mu)*y+204800*mu**4+(-163840*sig-192000)*mu**2)/((65536*sig**4+409600*sig**3+960000*sig**2+1000000*sig+390625)*y+(40960*sig**3+192000*sig**2+300000*sig+156250)*mu),(((20480*sig+32000)*mu**2-18432*sig**2-64000*sig-55000)*y+12800*mu**3+(-17920*sig-32000)*mu)/((4096*sig**3+19200*sig**2+30000*sig+15625)*y+(2560*sig**2+8000*sig+6250)*mu),(((73728000*sig+115200000)*mu**2+8192000*sig**2+38400000*sig+40000000)*y+(16384000*sig+71680000)*mu**3+(-16384000*sig**2-71680000*sig-64000000)*mu)/((16777216*sig**6+157286400*sig**5+614400000*sig**4+1280000000*sig**3+1500000000*sig**2+937500000*sig+244140625)*mu*y+(10485760*sig**5+81920000*sig**4+256000000*sig**3+400000000*sig**2+312500000*sig+97656250)*mu**2),(((36700160*sig**2+114688000*sig+89600000)*mu**3+(-15728640*sig**3-65536000*sig**2-89600000*sig-40000000)*mu)*y**2+((8388608*sig+13107200)*mu**6+(18874368*sig**2+108134400*sig+122880000)*mu**4+(-6291456*sig**3-101580800*sig**2-245760000*sig-160000000)*mu**2-20480000*sig**2-64000000*sig-50000000)*y+5242880*mu**7+(5242880*sig+24576000)*mu**5+(-10485760*sig**2-81920000*sig-89600000)*mu**3+(40960000*sig**2+115200000*sig+80000000)*mu)/((16777216*sig**6+157286400*sig**5+614400000*sig**4+1280000000*sig**3+1500000000*sig**2+937500000*sig+244140625)*mu*y+(10485760*sig**5+81920000*sig**4+256000000*sig**3+400000000*sig**2+312500000*sig+97656250)*mu**2),(((1310720*sig**2+4096000*sig+3200000)*mu**2+1024000*sig**2+3200000*sig+2500000)*y**2+((3932160*sig+6144000)*mu**5+(5505024*sig**2+23756800*sig+23680000)*mu**3+(-2359296*sig**3-13107200*sig**2-23680000*sig-14000000)*mu)*y+2457600*mu**6+(983040*sig+5120000)*mu**4+(-3112960*sig**2-12288000*sig-12000000)*mu**2)/((1048576*sig**5+8192000*sig**4+25600000*sig**3+40000000*sig**2+31250000*sig+9765625)*mu*y+(655360*sig**4+4096000*sig**3+9600000*sig**2+10000000*sig+3906250)*mu**2),(((20971520*sig+32768000)*mu**4+(10485760*sig**2+73728000*sig+89600000)*mu**2+8192000*sig**2+38400000*sig+40000000)*y**2+(-8388608*sig*mu**5+(-18874368*sig**2-131072000*sig-133120000)*mu**3+(6291456*sig**3+32768000*sig**2+20480000*sig-16000000)*mu)*y-5242880*mu**6+(-5242880*sig-57344000)*mu**4+(10485760*sig**2+65536000*sig+51200000)*mu**2)/((16777216*sig**6+157286400*sig**5+614400000*sig**4+1280000000*sig**3+1500000000*sig**2+937500000*sig+244140625)*mu*y+(10485760*sig**5+81920000*sig**4+256000000*sig**3+400000000*sig**2+312500000*sig+97656250)*mu**2)],[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,0,0,1,0]]])


cpdef inline cnp.ndarray[DTYPE_t, ndim=3] phi2(DTYPE_t[:] z):
    cdef DTYPE_t y = z[0]
    cdef DTYPE_t mu = z[1]
    cdef DTYPE_t sig = z[2]
    return np.array([[[((-1024*sig**2-3200*sig-2500)*mu*y**3+((-2048/5*sig**2-2560*sig-3000)*mu**2+2048/5*sig**3+1280*sig**2+1000*sig)*y**2+((-512*sig-1200)*mu**3+(2560*sig**2+6800*sig+4375)*mu)*y-160*mu**4+(1280*sig+1750)*mu**2+160*sig**2+250*sig)/((1024*sig**2+3200*sig+2500)*mu*y**2+(1280*sig+2000)*mu**2*y+400*mu**3+(400*sig+625)*mu),(((-24576/25*sig**2-3072*sig-2400)*mu**3+(24576/25*sig**3+5120*sig**2+8800*sig+5000)*mu)*y**2+((-6144/5*sig-1920)*mu**4+(18432/5*sig**2+12160*sig+10000)*mu**2)*y-384*mu**5+(1536*sig+2600)*mu**3+(-128*sig**2-200*sig)*mu)/((1024*sig**2+3200*sig+2500)*y**2+(1280*sig+2000)*mu*y+400*mu**2+400*sig+625),(((-16384/25*sig**3-3072*sig**2-4800*sig-2500)*mu**2+16384/25*sig**4+5120*sig**3+14400*sig**2+17500*sig+15625/2)*y**2+((-4096/5*sig**2-2560*sig-2000)*mu**3+(12288/5*sig**3+12800*sig**2+22000*sig+12500)*mu)*y+(-256*sig-400)*mu**4+(1024*sig**2+3600*sig+3125)*mu**2+256*sig**3+1600*sig**2+3125*sig+15625/8)/((1024*sig**2+3200*sig+2500)*y**2+(1280*sig+2000)*mu*y+400*mu**2+400*sig+625),(((262144/5*sig**2+163840*sig+128000)*mu**4+(-524288/5*sig**3-819200*sig**2-1792000*sig-1200000)*mu**2+262144/5*sig**4+327680*sig**3+640000*sig**2+400000*sig)*y**2+((65536*sig+102400)*mu**5+(-393216*sig**2-1638400*sig-1600000)*mu**3+(327680*sig**3+1945600*sig**2+3520000*sig+2000000)*mu)*y+20480*mu**6+(-184320*sig-416000)*mu**4+(471040*sig**2+1600000*sig+1300000)*mu**2+20480*sig**3+96000*sig**2+100000*sig)/((262144*sig**4+1638400*sig**3+3840000*sig**2+4000000*sig+1562500)*mu*y**2+(327680*sig**3+1536000*sig**2+2400000*sig+1250000)*mu**2*y+(102400*sig**2+320000*sig+250000)*mu**3+(102400*sig**3+480000*sig**2+750000*sig+390625)*mu),(((-524288/5*sig**3-491520*sig**2-768000*sig-400000)*mu**3+(524288/5*sig**4+491520*sig**3+768000*sig**2+400000*sig)*mu)*y**3+((-1048576/25*sig**2-131072*sig-102400)*mu**6+(2097152/25*sig**3+131072*sig**2-204800*sig-320000)*mu**4+(-1048576/25*sig**4+393216*sig**3+2150400*sig**2+3200000*sig+1500000)*mu**2-131072*sig**4-614400*sig**3-960000*sig**2-500000*sig)*y**2+((-262144/5*sig-81920)*mu**7+(1048576/5*sig**2+450560*sig+192000)*mu**5+(-786432/5*sig**3+81920*sig**2+1216000*sig+1100000)*mu**3+(-778240*sig**3-3456000*sig**2-5100000*sig-2500000)*mu)*y-16384*mu**8+(81920*sig+76800)*mu**6+(-114688*sig**2-102400*sig+80000)*mu**4+(49152*sig**3-332800*sig**2-1360000*sig-1125000)*mu**2-51200*sig**3-160000*sig**2-125000*sig)/((262144*sig**4+1638400*sig**3+3840000*sig**2+4000000*sig+1562500)*mu*y**2+(327680*sig**3+1536000*sig**2+2400000*sig+1250000)*mu**2*y+(102400*sig**2+320000*sig+250000)*mu**3+(102400*sig**3+480000*sig**2+750000*sig+390625)*mu),(((-32768/5*sig**3-30720*sig**2-48000*sig-25000)*mu**2+32768/5*sig**4+30720*sig**3+48000*sig**2+25000*sig)*y**3+((-393216/25*sig**2-49152*sig-38400)*mu**5+(262144/25*sig**3+24576*sig**2-20000)*mu**3+(131072/25*sig**4+73728*sig**3+268800*sig**2+380000*sig+187500)*mu)*y**2+((-98304/5*sig-30720)*mu**6+(262144/5*sig**2+140800*sig+92000)*mu**4+(98304/5*sig**3+153600*sig**2+348000*sig+243750)*mu**2+2560*sig**3+8000*sig**2+6250*sig)*y-6144*mu**7+(22528*sig+28800)*mu**5+(-2048*sig**2+12800*sig+30000)*mu**3+(2048*sig**3+9600*sig**2+20000*sig+15625)*mu)/((16384*sig**3+76800*sig**2+120000*sig+62500)*mu*y**2+(20480*sig**2+64000*sig+50000)*mu**2*y+(6400*sig+10000)*mu**3+(6400*sig**2+20000*sig+15625)*mu),(((-262144/5*sig**2-163840*sig-128000)*mu**4+(-163840*sig**2-512000*sig-400000)*mu**2+262144/5*sig**4+327680*sig**3+640000*sig**2+400000*sig)*y**3+((2097152/25*sig**2+196608*sig+102400)*mu**5+(393216*sig**2+1024000*sig+640000)*mu**3+(-2097152/25*sig**4-589824*sig**3-1126400*sig**2-640000*sig)*mu)*y**2+((524288/5*sig+143360)*mu**6+(-1048576/5*sig**2-184320*sig+160000)*mu**4+(-1572864/5*sig**3-2273280*sig**2-4352000*sig-2500000)*mu**2+20480*sig**3+96000*sig**2+100000*sig)*y+32768*mu**7-98304*sig*mu**5+(-32768*sig**2-512000*sig-560000)*mu**3+(-32768*sig**3-307200*sig**2-720000*sig-500000)*mu)/((262144*sig**4+1638400*sig**3+3840000*sig**2+4000000*sig+1562500)*mu*y**2+(327680*sig**3+1536000*sig**2+2400000*sig+1250000)*mu**2*y+(102400*sig**2+320000*sig+250000)*mu**3+(102400*sig**3+480000*sig**2+750000*sig+390625)*mu)],[(-24*mu**2+24*sig+50)/((16*sig+25)*mu),-y,0,(1024*mu**4+3200*mu**2+3072*sig**2+16000*sig+20000)/((4096*sig**3+19200*sig**2+30000*sig+15625)*mu),(160*mu**2-480*sig-1000)/((256*sig**2+800*sig+625)*mu),((-24*mu**2+24*sig+50)*y+40*mu)/((16*sig+25)*mu),((1024*mu**4+3200*mu**2+3072*sig**2+16000*sig+20000)*y-2560*mu**3+(-2560*sig-8000)*mu)/((4096*sig**3+19200*sig**2+30000*sig+15625)*mu)],[(256*mu**2-768*sig-1600)/(256*sig**2+800*sig+625),(20)/(16*sig+25),-y,(-16384*mu**4+32768*sig*mu**2-49152*sig**2-204800*sig-240000)/(65536*sig**4+409600*sig**3+960000*sig**2+1000000*sig+390625),(-2560*mu**2+7680*sig+16000)/(4096*sig**3+19200*sig**2+30000*sig+15625),((256*mu**2-768*sig-1600)*y-320*mu)/(256*sig**2+800*sig+625),((-16384*mu**4+32768*sig*mu**2-49152*sig**2-204800*sig-240000)*y+40960*mu**3-40960*sig*mu)/(65536*sig**4+409600*sig**3+960000*sig**2+1000000*sig+390625)],[(-4/5*sig**2-5/2*sig-125/64)/(mu),(-48/25*sig**2-6*sig-75/16)*mu,-32/25*sig**3-6*sig**2-75/8*sig-625/128,(-mu*y+2/5*mu**2-2/5*sig-5/4)/(mu),((-4/5*sig-5/4)*mu*y-8/25*mu**4+8/25*sig*mu**2+sig+25/16)/(mu),((-4/5*sig**2-5/2*sig-125/64)*y+(-48/25*sig-3)*mu**3+(-16/25*sig**2-3*sig-25/8)*mu)/(mu),((-2/5*mu**2-2/5*sig-5/4)*y+16/25*mu**3+(16/25*sig+2)*mu)/(mu)],[(-sig-25/16)/(mu),0,0,(-8*mu**2-8*sig-25)/((16*sig+25)*mu),(-mu*y+5/4)/(mu),((-sig-25/16)*y)/(mu),((-8*mu**2-8*sig-25)*y+20*mu)/((16*sig+25)*mu)],[1,0,0,0,0,0,0],[0,0,0,1,0,0,0]],[[(-24*mu**2+24*sig+50)/((16*sig+25)*mu),-y,0,(1024*mu**4+3200*mu**2+3072*sig**2+16000*sig+20000)/((4096*sig**3+19200*sig**2+30000*sig+15625)*mu),(160*mu**2-480*sig-1000)/((256*sig**2+800*sig+625)*mu),((-24*mu**2+24*sig+50)*y+40*mu)/((16*sig+25)*mu),((1024*mu**4+3200*mu**2+3072*sig**2+16000*sig+20000)*y-2560*mu**3+(-2560*sig-8000)*mu)/((4096*sig**3+19200*sig**2+30000*sig+15625)*mu)],[0,0,2,0,0,0,0],[((25600*sig+40000)*y+16000*mu)/((16384*sig**3+76800*sig**2+120000*sig+62500)*mu*y**2+(20480*sig**2+64000*sig+50000)*mu**2*y+(6400*sig+10000)*mu**3+(6400*sig**2+20000*sig+15625)*mu),(((-786432*sig**2-2457600*sig-1920000)*mu**2-524288*sig**3-2867200*sig**2-5120000*sig-3000000)*y**2+((-983040*sig-1536000)*mu**3+(-163840*sig**2-1024000*sig-1200000)*mu)*y-307200*mu**4+(-204800*sig-480000)*mu**2-307200*sig**2-1120000*sig-1000000)/((262144*sig**4+1638400*sig**3+3840000*sig**2+4000000*sig+1562500)*y**2+(327680*sig**3+1536000*sig**2+2400000*sig+1250000)*mu*y+(102400*sig**2+320000*sig+250000)*mu**2+102400*sig**3+480000*sig**2+750000*sig+390625),((-65536*sig**2-204800*sig-160000)*mu*y**2+((-81920*sig-128000)*mu**2+20480*sig**2+64000*sig+50000)*y-25600*mu**3+(-12800*sig-20000)*mu)/((16384*sig**3+76800*sig**2+120000*sig+62500)*y**2+(20480*sig**2+64000*sig+50000)*mu*y+(6400*sig+10000)*mu**2+6400*sig**2+20000*sig+15625),((-131072000*sig**2-409600000*sig-320000000)*mu*y**2+((-52428800*sig**2-327680000*sig-384000000)*mu**2+52428800*sig**3+327680000*sig**2+640000000*sig+400000000)*y+(-32768000*sig-102400000)*mu**3+(98304000*sig**2+307200000*sig+240000000)*mu)/((67108864*sig**6+629145600*sig**5+2457600000*sig**4+5120000000*sig**3+6000000000*sig**2+3750000000*sig+976562500)*mu*y**2+(83886080*sig**5+655360000*sig**4+2048000000*sig**3+3200000000*sig**2+2500000000*sig+781250000)*mu**2*y+(26214400*sig**4+163840000*sig**3+384000000*sig**2+400000000*sig+156250000)*mu**3+(26214400*sig**5+204800000*sig**4+640000000*sig**3+1000000000*sig**2+781250000*sig+244140625)*mu),((-83886080*sig**3-393216000*sig**2-614400000*sig-320000000)*mu**2*y**3+((-16777216*sig**2-52428800*sig-40960000)*mu**5+(-157286400*sig**2-491520000*sig-384000000)*mu**3+(50331648*sig**4+524288000*sig**3+1679360000*sig**2+2176000000*sig+1000000000)*mu)*y**2+((-20971520*sig-32768000)*mu**6+(20971520*sig**2-32768000*sig-102400000)*mu**4+(41943040*sig**3+458752000*sig**2+1075200000*sig+720000000)*mu**2-131072000*sig**3-614400000*sig**2-960000000*sig-500000000)*y-6553600*mu**7+(6553600*sig-10240000)*mu**5+(-6553600*sig**2+61440000*sig+96000000)*mu**3+(32768000*sig**3+92160000*sig**2+32000000*sig-50000000)*mu)/((67108864*sig**6+629145600*sig**5+2457600000*sig**4+5120000000*sig**3+6000000000*sig**2+3750000000*sig+976562500)*mu*y**2+(83886080*sig**5+655360000*sig**4+2048000000*sig**3+3200000000*sig**2+2500000000*sig+781250000)*mu**2*y+(26214400*sig**4+163840000*sig**3+384000000*sig**2+400000000*sig+156250000)*mu**3+(26214400*sig**5+204800000*sig**4+640000000*sig**3+1000000000*sig**2+781250000*sig+244140625)*mu),((-5242880*sig**3-24576000*sig**2-38400000*sig-20000000)*mu*y**3+((-8388608*sig**2-26214400*sig-20480000)*mu**4+(-12582912*sig**3-78643200*sig**2-153600000*sig-96000000)*mu**2+6553600*sig**3+30720000*sig**2+48000000*sig+25000000)*y**2+((-10485760*sig-16384000)*mu**5+(-7864320*sig**2-43008000*sig-48000000)*mu**3+(2621440*sig**3+22528000*sig**2+51200000*sig+35000000)*mu)*y-3276800*mu**6+(-3276800*sig-10240000)*mu**4+(-6553600*sig**2-20480000*sig-16000000)*mu**2)/((4194304*sig**5+32768000*sig**4+102400000*sig**3+160000000*sig**2+125000000*sig+39062500)*mu*y**2+(5242880*sig**4+32768000*sig**3+76800000*sig**2+80000000*sig+31250000)*mu**2*y+(1638400*sig**3+7680000*sig**2+12000000*sig+6250000)*mu**3+(1638400*sig**4+10240000*sig**3+24000000*sig**2+25000000*sig+9765625)*mu),(((-41943040*sig**2-131072000*sig-102400000)*mu**3+(-41943040*sig**3-262144000*sig**2-512000000*sig-320000000)*mu)*y**3+((33554432*sig**2+52428800*sig)*mu**4+(100663296*sig**3+629145600*sig**2+1146880000*sig+640000000)*mu**2+52428800*sig**3+327680000*sig**2+640000000*sig+400000000)*y**2+((41943040*sig+49152000)*mu**5+(83886080*sig**2+491520000*sig+537600000)*mu**3+(-41943040*sig**3-278528000*sig**2-486400000*sig-240000000)*mu)*y+13107200*mu**6+(26214400*sig+122880000)*mu**4+(39321600*sig**2+122880000*sig+128000000)*mu**2)/((67108864*sig**6+629145600*sig**5+2457600000*sig**4+5120000000*sig**3+6000000000*sig**2+3750000000*sig+976562500)*mu*y**2+(83886080*sig**5+655360000*sig**4+2048000000*sig**3+3200000000*sig**2+2500000000*sig+781250000)*mu**2*y+(26214400*sig**4+163840000*sig**3+384000000*sig**2+400000000*sig+156250000)*mu**3+(26214400*sig**5+204800000*sig**4+640000000*sig**3+1000000000*sig**2+781250000*sig+244140625)*mu)],[(-sig-25/16)/(mu),0,0,(-8*mu**2-8*sig-25)/((16*sig+25)*mu),(-mu*y+5/4)/(mu),((-sig-25/16)*y)/(mu),((-8*mu**2-8*sig-25)*y+20*mu)/((16*sig+25)*mu)],[0,0,0,0,0,2,0],[0,1,0,0,0,0,0],[0,0,0,0,1,0,0]],[[(256*mu**2-768*sig-1600)/(256*sig**2+800*sig+625),(20)/(16*sig+25),-y,(-16384*mu**4+32768*sig*mu**2-49152*sig**2-204800*sig-240000)/(65536*sig**4+409600*sig**3+960000*sig**2+1000000*sig+390625),(-2560*mu**2+7680*sig+16000)/(4096*sig**3+19200*sig**2+30000*sig+15625),((256*mu**2-768*sig-1600)*y-320*mu)/(256*sig**2+800*sig+625),((-16384*mu**4+32768*sig*mu**2-49152*sig**2-204800*sig-240000)*y+40960*mu**3-40960*sig*mu)/(65536*sig**4+409600*sig**3+960000*sig**2+1000000*sig+390625)],[((25600*sig+40000)*y+16000*mu)/((16384*sig**3+76800*sig**2+120000*sig+62500)*mu*y**2+(20480*sig**2+64000*sig+50000)*mu**2*y+(6400*sig+10000)*mu**3+(6400*sig**2+20000*sig+15625)*mu),(((-786432*sig**2-2457600*sig-1920000)*mu**2-524288*sig**3-2867200*sig**2-5120000*sig-3000000)*y**2+((-983040*sig-1536000)*mu**3+(-163840*sig**2-1024000*sig-1200000)*mu)*y-307200*mu**4+(-204800*sig-480000)*mu**2-307200*sig**2-1120000*sig-1000000)/((262144*sig**4+1638400*sig**3+3840000*sig**2+4000000*sig+1562500)*y**2+(327680*sig**3+1536000*sig**2+2400000*sig+1250000)*mu*y+(102400*sig**2+320000*sig+250000)*mu**2+102400*sig**3+480000*sig**2+750000*sig+390625),((-65536*sig**2-204800*sig-160000)*mu*y**2+((-81920*sig-128000)*mu**2+20480*sig**2+64000*sig+50000)*y-25600*mu**3+(-12800*sig-20000)*mu)/((16384*sig**3+76800*sig**2+120000*sig+62500)*y**2+(20480*sig**2+64000*sig+50000)*mu*y+(6400*sig+10000)*mu**2+6400*sig**2+20000*sig+15625),((-131072000*sig**2-409600000*sig-320000000)*mu*y**2+((-52428800*sig**2-327680000*sig-384000000)*mu**2+52428800*sig**3+327680000*sig**2+640000000*sig+400000000)*y+(-32768000*sig-102400000)*mu**3+(98304000*sig**2+307200000*sig+240000000)*mu)/((67108864*sig**6+629145600*sig**5+2457600000*sig**4+5120000000*sig**3+6000000000*sig**2+3750000000*sig+976562500)*mu*y**2+(83886080*sig**5+655360000*sig**4+2048000000*sig**3+3200000000*sig**2+2500000000*sig+781250000)*mu**2*y+(26214400*sig**4+163840000*sig**3+384000000*sig**2+400000000*sig+156250000)*mu**3+(26214400*sig**5+204800000*sig**4+640000000*sig**3+1000000000*sig**2+781250000*sig+244140625)*mu),((-83886080*sig**3-393216000*sig**2-614400000*sig-320000000)*mu**2*y**3+((-16777216*sig**2-52428800*sig-40960000)*mu**5+(-157286400*sig**2-491520000*sig-384000000)*mu**3+(50331648*sig**4+524288000*sig**3+1679360000*sig**2+2176000000*sig+1000000000)*mu)*y**2+((-20971520*sig-32768000)*mu**6+(20971520*sig**2-32768000*sig-102400000)*mu**4+(41943040*sig**3+458752000*sig**2+1075200000*sig+720000000)*mu**2-131072000*sig**3-614400000*sig**2-960000000*sig-500000000)*y-6553600*mu**7+(6553600*sig-10240000)*mu**5+(-6553600*sig**2+61440000*sig+96000000)*mu**3+(32768000*sig**3+92160000*sig**2+32000000*sig-50000000)*mu)/((67108864*sig**6+629145600*sig**5+2457600000*sig**4+5120000000*sig**3+6000000000*sig**2+3750000000*sig+976562500)*mu*y**2+(83886080*sig**5+655360000*sig**4+2048000000*sig**3+3200000000*sig**2+2500000000*sig+781250000)*mu**2*y+(26214400*sig**4+163840000*sig**3+384000000*sig**2+400000000*sig+156250000)*mu**3+(26214400*sig**5+204800000*sig**4+640000000*sig**3+1000000000*sig**2+781250000*sig+244140625)*mu),((-5242880*sig**3-24576000*sig**2-38400000*sig-20000000)*mu*y**3+((-8388608*sig**2-26214400*sig-20480000)*mu**4+(-12582912*sig**3-78643200*sig**2-153600000*sig-96000000)*mu**2+6553600*sig**3+30720000*sig**2+48000000*sig+25000000)*y**2+((-10485760*sig-16384000)*mu**5+(-7864320*sig**2-43008000*sig-48000000)*mu**3+(2621440*sig**3+22528000*sig**2+51200000*sig+35000000)*mu)*y-3276800*mu**6+(-3276800*sig-10240000)*mu**4+(-6553600*sig**2-20480000*sig-16000000)*mu**2)/((4194304*sig**5+32768000*sig**4+102400000*sig**3+160000000*sig**2+125000000*sig+39062500)*mu*y**2+(5242880*sig**4+32768000*sig**3+76800000*sig**2+80000000*sig+31250000)*mu**2*y+(1638400*sig**3+7680000*sig**2+12000000*sig+6250000)*mu**3+(1638400*sig**4+10240000*sig**3+24000000*sig**2+25000000*sig+9765625)*mu),(((-41943040*sig**2-131072000*sig-102400000)*mu**3+(-41943040*sig**3-262144000*sig**2-512000000*sig-320000000)*mu)*y**3+((33554432*sig**2+52428800*sig)*mu**4+(100663296*sig**3+629145600*sig**2+1146880000*sig+640000000)*mu**2+52428800*sig**3+327680000*sig**2+640000000*sig+400000000)*y**2+((41943040*sig+49152000)*mu**5+(83886080*sig**2+491520000*sig+537600000)*mu**3+(-41943040*sig**3-278528000*sig**2-486400000*sig-240000000)*mu)*y+13107200*mu**6+(26214400*sig+122880000)*mu**4+(39321600*sig**2+122880000*sig+128000000)*mu**2)/((67108864*sig**6+629145600*sig**5+2457600000*sig**4+5120000000*sig**3+6000000000*sig**2+3750000000*sig+976562500)*mu*y**2+(83886080*sig**5+655360000*sig**4+2048000000*sig**3+3200000000*sig**2+2500000000*sig+781250000)*mu**2*y+(26214400*sig**4+163840000*sig**3+384000000*sig**2+400000000*sig+156250000)*mu**3+(26214400*sig**5+204800000*sig**4+640000000*sig**3+1000000000*sig**2+781250000*sig+244140625)*mu)],[((4096000*sig**2+12800000*sig+10000000)*y**2+(-16384000*sig**2-46080000*sig-32000000)*mu*y+(-10240000*sig-14400000)*mu**2+1600000*sig+2500000)/((4194304*sig**5+32768000*sig**4+102400000*sig**3+160000000*sig**2+125000000*sig+39062500)*mu*y**2+(5242880*sig**4+32768000*sig**3+76800000*sig**2+80000000*sig+31250000)*mu**2*y+(1638400*sig**3+7680000*sig**2+12000000*sig+6250000)*mu**3+(1638400*sig**4+10240000*sig**3+24000000*sig**2+25000000*sig+9765625)*mu),((-2621440*sig**3-12288000*sig**2-19200000*sig-10000000)*y**3+((20971520*sig**2+65536000*sig+51200000)*mu**3+(-2097152*sig**3-6553600*sig**2-5120000*sig)*mu)*y**2+((26214400*sig+40960000)*mu**4+(-22282240*sig**2-62464000*sig-43200000)*mu**2-1024000*sig**2-3200000*sig-2500000)*y+8192000*mu**5+(-4915200*sig-5120000)*mu**3+(3276800*sig**2+12800000*sig+12000000)*mu)/((4194304*sig**5+32768000*sig**4+102400000*sig**3+160000000*sig**2+125000000*sig+39062500)*y**2+(5242880*sig**4+32768000*sig**3+76800000*sig**2+80000000*sig+31250000)*mu*y+(1638400*sig**3+7680000*sig**2+12000000*sig+6250000)*mu**2+1638400*sig**4+10240000*sig**3+24000000*sig**2+25000000*sig+9765625),(((1310720*sig**2+4096000*sig+3200000)*mu**2-1048576*sig**3-5324800*sig**2-8960000*sig-5000000)*y**2+((1638400*sig+2560000)*mu**3+(-2129920*sig**2-7168000*sig-6000000)*mu)*y+512000*mu**4+(-409600*sig-800000)*mu**2-409600*sig**2-1440000*sig-1250000)/((262144*sig**4+1638400*sig**3+3840000*sig**2+4000000*sig+1562500)*y**2+(327680*sig**3+1536000*sig**2+2400000*sig+1250000)*mu*y+(102400*sig**2+320000*sig+250000)*mu**2+102400*sig**3+480000*sig**2+750000*sig+390625),(((4718592000*sig**2+14745600000*sig+11520000000)*mu**2+524288000*sig**3+3276800000*sig**2+6400000000*sig+4000000000)*y**2+((2097152000*sig**2+12451840000*sig+14336000000)*mu**3+(-2097152000*sig**3-12451840000*sig**2-22528000000*sig-12800000000)*mu)*y+(1310720000*sig+3891200000)*mu**4+(-3932160000*sig**2-12288000000*sig-9280000000)*mu**2+204800000*sig**2+960000000*sig+1000000000)/((1073741824*sig**7+11744051200*sig**6+55050240000*sig**5+143360000000*sig**4+224000000000*sig**3+210000000000*sig**2+109375000000*sig+24414062500)*mu*y**2+(1342177280*sig**6+12582912000*sig**5+49152000000*sig**4+102400000000*sig**3+120000000000*sig**2+75000000000*sig+19531250000)*mu**2*y+(419430400*sig**5+3276800000*sig**4+10240000000*sig**3+16000000000*sig**2+12500000000*sig+3906250000)*mu**3+(419430400*sig**6+3932160000*sig**5+15360000000*sig**4+32000000000*sig**3+37500000000*sig**2+23437500000*sig+6103515625)*mu),(((2348810240*sig**3+11010048000*sig**2+17203200000*sig+8960000000)*mu**3+(-1006632960*sig**4-5767168000*sig**3-12288000000*sig**2-11520000000*sig-4000000000)*mu)*y**3+((536870912*sig**2+1677721600*sig+1310720000)*mu**6+(-268435456*sig**3+3355443200*sig**2+12451840000*sig+10240000000)*mu**4+(-805306368*sig**4-12582912000*sig**3-45219840000*sig**2-61440000000*sig-28800000000)*mu**2-1310720000*sig**3-6144000000*sig**2-9600000000*sig-5000000000)*y**2+((671088640*sig+1048576000)*mu**7+(-1174405120*sig**2-655360000*sig+1843200000)*mu**5+(-167772160*sig**3-9437184000*sig**2-26419200000*sig-18880000000)*mu**3+(4849664000*sig**3+21299200000*sig**2+31040000000*sig+15000000000)*mu)*y+209715200*mu**8-419430400*sig*mu**6+(629145600*sig**2-655360000*sig-2048000000)*mu**4+(-838860800*sig**3-655360000*sig**2+4096000000*sig+4800000000)*mu**2-512000000*sig**2-1600000000*sig-1250000000)/((1073741824*sig**7+11744051200*sig**6+55050240000*sig**5+143360000000*sig**4+224000000000*sig**3+210000000000*sig**2+109375000000*sig+24414062500)*mu*y**2+(1342177280*sig**6+12582912000*sig**5+49152000000*sig**4+102400000000*sig**3+120000000000*sig**2+75000000000*sig+19531250000)*mu**2*y+(419430400*sig**5+3276800000*sig**4+10240000000*sig**3+16000000000*sig**2+12500000000*sig+3906250000)*mu**3+(419430400*sig**6+3932160000*sig**5+15360000000*sig**4+32000000000*sig**3+37500000000*sig**2+23437500000*sig+6103515625)*mu),(((83886080*sig**3+393216000*sig**2+614400000*sig+320000000)*mu**2+65536000*sig**3+307200000*sig**2+480000000*sig+250000000)*y**3+((251658240*sig**2+786432000*sig+614400000)*mu**5+(201326592*sig**3+1415577600*sig**2+2949120000*sig+1920000000)*mu**3+(-50331648*sig**4-524288000*sig**3-1679360000*sig**2-2176000000*sig-1000000000)*mu)*y**2+((314572800*sig+491520000)*mu**6+(-62914560*sig**2+294912000*sig+614400000)*mu**4+(-167772160*sig**3-1179648000*sig**2-2483200000*sig-1640000000)*mu**2+25600000*sig**2+80000000*sig+62500000)*y+98304000*mu**7+(-19660800*sig+112640000)*mu**5+(124518400*sig**2+348160000*sig+224000000)*mu**3+(-19660800*sig**3-71680000*sig**2-96000000*sig-50000000)*mu)/((67108864*sig**6+629145600*sig**5+2457600000*sig**4+5120000000*sig**3+6000000000*sig**2+3750000000*sig+976562500)*mu*y**2+(83886080*sig**5+655360000*sig**4+2048000000*sig**3+3200000000*sig**2+2500000000*sig+781250000)*mu**2*y+(26214400*sig**4+163840000*sig**3+384000000*sig**2+400000000*sig+156250000)*mu**3+(26214400*sig**5+204800000*sig**4+640000000*sig**3+1000000000*sig**2+781250000*sig+244140625)*mu),(((1342177280*sig**2+4194304000*sig+3276800000)*mu**4+(671088640*sig**3+5767168000*sig**2+13107200000*sig+8960000000)*mu**2+524288000*sig**3+3276800000*sig**2+6400000000*sig+4000000000)*y**3+((-1073741824*sig**2-1677721600*sig)*mu**5+(-2415919104*sig**3-17196646400*sig**2-32768000000*sig-18432000000)*mu**3+(805306368*sig**4+4613734400*sig**3+6553600000*sig**2-3200000000)*mu)*y**2+((-1342177280*sig-1572864000)*mu**6+(-1342177280*sig**2-11796480000*sig-14131200000)*mu**4+(2684354560*sig**3+17563648000*sig**2+31539200000*sig+16960000000)*mu**2+204800000*sig**2+960000000*sig+1000000000)*y-419430400*mu**7+(-314572800*sig-3112960000)*mu**5+(-629145600*sig**2-655360000*sig-1024000000)*mu**3+(314572800*sig**3+2129920000*sig**2+4096000000*sig+2400000000)*mu)/((1073741824*sig**7+11744051200*sig**6+55050240000*sig**5+143360000000*sig**4+224000000000*sig**3+210000000000*sig**2+109375000000*sig+24414062500)*mu*y**2+(1342177280*sig**6+12582912000*sig**5+49152000000*sig**4+102400000000*sig**3+120000000000*sig**2+75000000000*sig+19531250000)*mu**2*y+(419430400*sig**5+3276800000*sig**4+10240000000*sig**3+16000000000*sig**2+12500000000*sig+3906250000)*mu**3+(419430400*sig**6+3932160000*sig**5+15360000000*sig**4+32000000000*sig**3+37500000000*sig**2+23437500000*sig+6103515625)*mu)],[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,0,0,1,0]]])
