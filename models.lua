require 'nngraph'

local SpatialBatchNormalization = nn.SpatialBatchNormalization
local SpatialConvolution = nn.SpatialConvolution
local SpatialFullConvolution = nn.SpatialFullConvolution
function defineD_JointProb_U(input_nc,ndf,nz)
    -- The first input is Z we upsample that to get to the image size/2 ,the seocnd input is image
    local netD= nil

    local z0=-nn.Identity()
    local z1=z0-SpatialFullConvolution(nz, ndf * 8, 4, 4)- SpatialBatchNormalization(ndf * 8)

-- state size: (ndf*8) x 4 x 4
    local z2= z1-nn.LeakyReLU(0.2, true) - SpatialFullConvolution(ndf * 8, ndf * 4, 4, 4, 2, 2, 1, 1) -SpatialBatchNormalization(ndf * 4)
-- state size: (ndf*4) x 8 x 8
    local z3=z2 - nn.LeakyReLU(0.2, true) - SpatialFullConvolution(ndf * 4, ndf * 2, 4, 4, 2, 2, 1, 1) - SpatialBatchNormalization(ndf * 2)
-- state size: (ndf*2) x 16 x 16
    local z4= z3 - nn.LeakyReLU(0.2, true) - SpatialFullConvolution(ndf * 2, ndf, 4, 4, 2, 2, 1, 1) - SpatialBatchNormalization(ndf)
-- state size: (ndf) x 32 x 32
    local z5= z4 - nn.LeakyReLU(0.2, true) - SpatialFullConvolution(ndf, input_nc, 4, 4, 2, 2, 1, 1) - nn.Tanh()
-- stae size = nc x 64 x 64

    local d0_=-nn.Identity()

    local d0 = {d0_,z5} - nn.JoinTable(2)
-- state size : (nc*2) x 64 x 64
    local d1_ = d0 - nn.SpatialConvolution(input_nc*2, ndf, 4, 4, 2, 2, 1, 1) - SpatialBatchNormalization(ndf)
    local d1= {d1_,z4} - nn.JoinTable(2)

--  state size ; (ndf*2)*32 x 32
    local d2_= d1 -nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ndf*2, ndf*4, 4, 4, 2, 2, 1, 1) -SpatialBatchNormalization (ndf*4)
    local d2= {d2_,z3} -nn.JoinTable(2)

--  state size ; (ndf*4+ndf*2)x16 x 16

    local d3_= d2 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ndf*6, ndf*8, 4, 4, 2, 2, 1, 1) -SpatialBatchNormalization(ndf*8)
    local d3= {d3_,z2} -nn.JoinTable(2)

-- state size ; (ndf*8+ndf*4) x 8 x 8
    local d4_= d3 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ndf*12, ndf*12, 4, 4, 2, 2, 1, 1) -SpatialBatchNormalization(ndf*12)
    local d4= {d4_,z1} -nn.JoinTable(2)
-- state size ; (ndf*12+ndf*8) x 4 x 4
    local d5= d4 -nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ndf*20, ndf*5, 4, 4, 2, 2, 1, 1) -SpatialBatchNormalization(ndf*5)
-- state size ; (ndf*5) x 2 x 2
    local d6= d5 -nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ndf*5,nz, 4, 4, 2, 2, 1, 1) -- -SpatialBatchNormalization(nz)
    local d7= {d6,z0} - nn.JoinTable(2)
    local d8= d7 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(2*nz,1,1,1) - nn.LeakyReLU(0.2, true) -nn.View(1):setNumInputDims(3)
    --local d8= d7 - nn.ReLU(true) - nn.SpatialConvolution(2*nz,1,1,1) - nn.LogSigmoid() -nn.MulConstant(-1,false)
   -- local d8= d7 - nn.ReLU(true) - nn.SpatialConvolution(2*nz,1,1,1) - nn.Sigmoid() - nn.AddConstant(0.0000001,false)-nn.Log() - nn.MulConstant(-1,false)


    netD = nn.gModule({z0,d0_},{d8})

    return netD

end

function defineD32_JointProb_U_BiGAN(input_nc,ndf,nz)
    -- The first input is Z we upsample that to get to the image size/2 ,the seocnd input is image
    local netD= nil

    local z0=-nn.Identity()
    local z1=z0-SpatialFullConvolution(nz, ndf * 16, 4, 4)- SpatialBatchNormalization(ndf * 16) -nn.LeakyReLU(0.2,true)

-- state size: (16*ndf) x 4 x 4
    local z2= z1- SpatialFullConvolution(ndf * 16, ndf * 8, 4, 4, 2, 2, 1, 1) -SpatialBatchNormalization(ndf * 8) -nn.LeakyReLU(0.2,true)
-- state size: (ndf*8) x 8 x 8
    local z3=z2  - SpatialFullConvolution(ndf * 8, ndf *8, 4, 4, 2, 2, 1, 1) - SpatialBatchNormalization(ndf * 8) -nn.LeakyReLU(0.2,true)
-- state size: (ndf*8) x 16 x 16
   -- local z4= z3 - nn.LeakyReLU(0.2, true) - SpatialFullConvolution(ndf * 2, ndf, 4, 4, 2, 2, 1, 1) - SpatialBatchNormalization(ndf)
-- state size: (ndf) x 32 x 32
  --  local z5= z4 - nn.LeakyReLU(0.2, true) - SpatialFullConvolution(ndf, input_nc, 4, 4, 2, 2, 1, 1) - nn.Tanh()
-- stae size = nc x 64 x 64

    local d0=-nn.Identity()

    --local d0 = {d0_,z5} - nn.JoinTable(2)
-- state size : (nc) x 32 x 32
    local d1_=d0 -nn.SpatialConvolution(input_nc, ndf, 3, 3, 1, 1, 1, 1) - SpatialBatchNormalization(ndf) -nn.LeakyReLU(0.2,true)
    local d1 = d1_ - nn.SpatialConvolution(ndf, ndf*8, 4, 4, 2, 2, 1, 1) - SpatialBatchNormalization(ndf) -nn.LeakyReLU(0.2,true)


--  state size ; (ndf*8)*16 x 16
    local d2_0={d1,z3} -nn.CAddTable()
    local d2_1= d2_0 -nn.SpatialConvolution(ndf*8, ndf*8, 3, 3, 1, 1, 1, 1) - SpatialBatchNormalization(ndf*8) -nn.LeakyReLU(0.2,true)
    local d2_2= d2_1 - nn.SpatialConvolution(ndf*8, ndf*8, 4, 4, 2, 2, 1, 1) - SpatialBatchNormalization(ndf*8) -nn.LeakyReLU(0.2,true)


--  state size ; (ndf*8)x 8 x 8
    local d3_0={d2_2,z2} -nn.CAddTable()
    local d3_1=d3_0 -nn.SpatialConvolution(ndf*8, ndf*8, 3, 3, 1, 1, 1, 1) - SpatialBatchNormalization(ndf*8) -nn.LeakyReLU(0.2,true)
    local d3_2= d3_1 - nn.SpatialConvolution(ndf*8, ndf*16, 4, 4, 2, 2, 1, 1) -SpatialBatchNormalization(ndf*16) -nn.LeakyReLU(0.2,true)

-- state size ; (ndf*16) x 4 x 4
    local d4_0={d3_2,z1} -nn.CAddTable()
    local d4_1= d4_0 -nn.SpatialConvolution(ndf*16, ndf*16, 3, 3, 1, 1, 1, 1) - SpatialBatchNormalization(ndf*16) -nn.LeakyReLU(0.2,true)
    local d4_2=d4_1 -nn.SpatialConvolution(ndf*16, ndf*16, 4, 4, 2, 2, 1, 1) -SpatialBatchNormalization(ndf*16) -nn.LeakyReLU(0.2,true)

-- state size ; (ndf*16) x 2 x 2
    local d5_0= d4_2 - nn.SpatialConvolution(ndf*16, ndf*16, 2, 2) -SpatialBatchNormalization(ndf*16) -nn.LeakyReLU(0.2,true)--nn.View(1):setNumInputDims(3)
    local d5_1=d5_0 - nn.SpatialConvolution(ndf*16, 1, 1, 1)  -nn.LeakyReLU(0.2,true)--nn.View(1):setNumInputDims(3)
-- state size ; (ndf*16) x 1 x 1

   -- local d8= d7 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(2*nz,1,1,1) - nn.LeakyReLU(0.2, true) -nn.View(1):setNumInputDims(3)
    --local d8= d7 - nn.ReLU(true) - nn.SpatialConvolution(2*nz,1,1,1) - nn.LogSigmoid() -nn.MulConstant(-1,false)
   -- local d8= d7 - nn.ReLU(true) - nn.SpatialConvolution(2*nz,1,1,1) - nn.Sigmoid() - nn.AddConstant(0.0000001,false)-nn.Log() - nn.MulConstant(-1,false)


    netD = nn.gModule({z0,d0},{d5_1})

    return netD

end

function defineD_JointProb_U_sub(input_nc,ndf,nz)
    -- The first input is Z we upsample that to get to the image size/2 ,the seocnd input is image
    local netD= nil

    local z0=-nn.Identity()
    local z1=z0-SpatialFullConvolution(nz, ndf * 8, 4, 4)- SpatialBatchNormalization(ndf * 8)-nn.ReLU( true)

-- state size: (ndf*8) x 4 x 4
    local z2= z1 - SpatialFullConvolution(ndf * 8, ndf * 4, 4, 4, 2, 2, 1, 1) -SpatialBatchNormalization(ndf * 4) - nn.ReLU( true)
-- state size: (ndf*4) x 8 x 8
    local z3=z2  - SpatialFullConvolution(ndf * 4, ndf * 2, 4, 4, 2, 2, 1, 1) - SpatialBatchNormalization(ndf * 2) - nn.ReLU( true)
-- state size: (ndf*2) x 16 x 16
    local z4= z3- SpatialFullConvolution(ndf * 2, ndf, 4, 4, 2, 2, 1, 1) - SpatialBatchNormalization(ndf) - nn.ReLU( true)
-- state size: (ndf) x 32 x 32
    local z5= z4  - SpatialFullConvolution(ndf, input_nc, 4, 4, 2, 2, 1, 1) - nn.Tanh()
-- stae size = nc x 64 x 64

    local d0_= -nn.Identity()

    local d0 = {d0_,z5} - nn.CSubTable() -nn.Mean(4) -nn.Mean(3)-nn.Mean(2)--nn.View(bs)
    --print(d0)
-- state size : (nc*2) x 64 x 64

    local d1_ = d0_ - nn.SpatialConvolution(input_nc, ndf, 4, 4, 2, 2, 1, 1) - SpatialBatchNormalization(ndf) -nn.ReLU(true)
--  state size ; (ndf)*32 x 32
    local d1= {d1_,z4} - nn.CSubTable() - nn.Mean(4) -nn.Mean(3)-nn.Mean(2) --nn.View(bs)
   -- print(d1)

    local d2_= d1_  - nn.SpatialConvolution(ndf, ndf*2, 4, 4, 2, 2, 1, 1) -SpatialBatchNormalization (ndf*2) -nn.ReLU(true)

--  state size ; (ndf*2)x16 x 16
    local d2= {d2_,z3} -nn.CSubTable()  - nn.Mean(4) -nn.Mean(3)-nn.Mean(2)--nn.View(bs)
   -- print(d2)
    local d3_= d2_  - nn.SpatialConvolution(ndf*2, ndf*4, 4, 4, 2, 2, 1, 1) -SpatialBatchNormalization(ndf*4) - nn.ReLU( true)
-- state size ; (ndf*4) x 8 x 8
    local d3= {d3_,z2} -nn.CSubTable() - nn.Mean(4) -nn.Mean(3)-nn.Mean(2)--nn.View(bs)
   -- print(d3)
    local d4_= d3_  - nn.SpatialConvolution(ndf*4, ndf*8, 4, 4, 2, 2, 1, 1) -SpatialBatchNormalization(ndf*8) - nn.ReLU( true)
-- state size ; (ndf*8) x 4 x 4
    local d4= {d4_,z1} -nn.CSubTable()  - nn.Mean(4) -nn.Mean(3)-nn.Mean(2)--nn.View(bs)
    --print(d4)
--]]
--[[
    local d5= d4 -nn.ReLU( true) - nn.SpatialConvolution(ndf*20, ndf*5, 4, 4, 2, 2, 1, 1) -SpatialBatchNormalization(ndf*5)
-- state size ; (ndf*5) x 2 x 2
    local d6= d5 - nn.ReLU( true) - nn.SpatialConvolution(ndf*5,nz, 4, 4, 2, 2, 1, 1) -- -SpatialBatchNormalization(nz)
    local d7= {d6,z0} - nn.JoinTable(2)
    local d8= d7 - nn.ReLU( true) - nn.SpatialConvolution(2*nz,1,1,1) - nn.ReLU(true) -nn.View(1):setNumInputDims(3)
    --local d8= d7 - nn.ReLU(true) - nn.SpatialConvolution(2*nz,1,1,1) - nn.LogSigmoid() -nn.MulConstant(-1,false)
   -- local d8= d7 - nn.ReLU(true) - nn.SpatialConvolution(2*nz,1,1,1) - nn.Sigmoid() - nn.AddConstant(0.0000001,false)-nn.Log() - nn.MulConstant(-1,false)


--]]
    --local d8=nn.JoinTable(dimension, nInputDims)
   -- local d8={d0,d1} -CAddTable() --CAddTable()
   -- local d9={d8,d2} -CAddTable()

    --local d8={d0,d1,d2,d3,d4} - nn.JoinTable(2) -nn.Mean() --CAddTable()    --
    local d8={d0,d4,d1,d2,d3}  -nn.CAddTable() -nn.Abs() --nn.View(bs,1)
    netD = nn.gModule({z0,d0_},{d8})

    return netD

end


function defineD_JointProb_U_Zsum(input_nc,ndf,nz,bs)
    -- The first input is Z we upsample that to get to the image size/2 ,the seocnd input is image
    local netD= nil

    local z0=-nn.Identity()
    local z1=z0 -nn.View(bs,nz)-nn.Linear(nz,1024)-nn.Dropout(0.2)-nn.ReLU( true)
    local z2=z1 -nn.Linear(1024,1024) -nn.Dropout(0.2) -nn.ReLU(true)


    local d0= -nn.Identity()


-- state size : (nc*2) x 64 x 64

    local d1 = d0 - nn.SpatialConvolution(input_nc, ndf, 4, 4, 2, 2, 1, 1) - SpatialBatchNormalization(ndf) -nn.ReLU(true)
--  state size ; (ndf)*32 x 32

   -- print(d1)

    local d2= d1  - nn.SpatialConvolution(ndf, ndf*2, 4, 4, 2, 2, 1, 1) -SpatialBatchNormalization (ndf*2) -nn.ReLU(true)

--  state size ; (ndf*2)x16 x 16

   -- print(d2)
    local d3= d2  - nn.SpatialConvolution(ndf*2, ndf*4, 4, 4, 2, 2, 1, 1) -SpatialBatchNormalization(ndf*4) - nn.ReLU( true)
-- state size ; (ndf*4) x 8 x 8

   -- print(d3)
    local d4= d3  - nn.SpatialConvolution(ndf*4, ndf*8, 4, 4, 2, 2, 1, 1) -SpatialBatchNormalization(ndf*8) - nn.ReLU( true)

-- state size ; (ndf*8) x 4 x 4
   local d5= d4 - nn.SpatialConvolution(ndf*8, ndf*8, 4, 4, 2, 2, 1, 1) -SpatialBatchNormalization(ndf*8) -nn.ReLU(true) --nn.View(torch.LongStorage{bs,ndf*8 *2*2})

--state size ; (ndf*8) x 2x 2
    local d6= d5 -nn.View(torch.LongStorage{bs,ndf*8 *2*2}) -nn.Linear((ndf*8*2*2),1024) -nn.ReLU(true)

    --print(d4)


    local d8={d6,z2} -nn.CSubTable() -nn.Abs() -nn.Mean(2)
    netD = nn.gModule({z0,d0},{d8})
    --netD = nn.gModule({d0},{d6})

    return netD

end

function defineD_JointProb_U_Zsum_lowleak(input_nc,ndf,nz,bs)
    -- The first input is Z we upsample that to get to the image size/2 ,the seocnd input is image
    local netD= nil

    local z0=-nn.Identity()
    local z1=z0 -nn.View(bs,nz)-nn.Linear(nz,1024)-nn.Dropout(0.2)-nn.LeakyReLU( true)
    local z2=z1 -nn.Linear(1024,1024) -nn.Dropout(0.2) -nn.LeakyReLU(true)


    local d0= -nn.Identity()


-- state size : (nc*2) x 64 x 64

    local d1 = d0 - nn.SpatialConvolution(input_nc, ndf, 4, 4, 2, 2, 1, 1) - SpatialBatchNormalization(ndf) -nn.LeakyReLU(0.02,true)
--  state size ; (ndf)*32 x 32

   -- print(d1)

    local d2= d1  - nn.SpatialConvolution(ndf, ndf*2, 4, 4, 2, 2, 1, 1) -SpatialBatchNormalization (ndf*2) -nn.LeakyReLU(0.02,true)

--  state size ; (ndf*2)x16 x 16

   -- print(d2)
    local d3= d2  - nn.SpatialConvolution(ndf*2, ndf*4, 4, 4, 2, 2, 1, 1) -SpatialBatchNormalization(ndf*4) - nn.LeakyReLU( 0.02,true)
-- state size ; (ndf*4) x 8 x 8

   -- print(d3)
    local d4= d3  - nn.SpatialConvolution(ndf*4, ndf*8, 4, 4, 2, 2, 1, 1) -SpatialBatchNormalization(ndf*8) - nn.LeakyReLU(0.02, true)

-- state size ; (ndf*8) x 4 x 4
   local d5= d4 - nn.SpatialConvolution(ndf*8, ndf*8, 4, 4, 2, 2, 1, 1) -SpatialBatchNormalization(ndf*8) -nn.LeakyReLU(0.02,true) --nn.View(torch.LongStorage{bs,ndf*8 *2*2})

--state size ; (ndf*8) x 2x 2
    local d6= d5 -nn.View(torch.LongStorage{bs,ndf*8 *2*2}) -nn.Linear((ndf*8*2*2),1024) -nn.LeakyReLU(0.02,true)

    --print(d4)


    local d8={d6,z2} -nn.CSubTable() -nn.Abs() -nn.Mean(2)
    netD = nn.gModule({z0,d0},{d8})
    --netD = nn.gModule({d0},{d6})

    return netD

end

function defineD_JointProb_U_Zsum_lowleak_droplast(input_nc,ndf,nz,bs)
    -- The first input is Z we upsample that to get to the image size/2 ,the seocnd input is image
    local netD= nil

    local z0=-nn.Identity()
    local z1=z0 -nn.View(bs,nz)-nn.Linear(nz,1024)-nn.Dropout(0.2)-nn.LeakyReLU( true) -- is equalt to inplace true and leak =0.01
    local z2=z1 -nn.Linear(1024,1024) -nn.Dropout(0.2) -nn.LeakyReLU(true)


    local d0= -nn.Identity()


-- state size : (nc*2) x 64 x 64

    local d1 = d0 - nn.SpatialConvolution(input_nc, ndf, 4, 4, 2, 2, 1, 1) - SpatialBatchNormalization(ndf) -nn.LeakyReLU(0.02,true)
--  state size ; (ndf)*32 x 32

   -- print(d1)

    local d2= d1  - nn.SpatialConvolution(ndf, ndf*2, 4, 4, 2, 2, 1, 1) -SpatialBatchNormalization (ndf*2) -nn.LeakyReLU(0.02,true)

--  state size ; (ndf*2)x16 x 16

   -- print(d2)
    local d3= d2  - nn.SpatialConvolution(ndf*2, ndf*4, 4, 4, 2, 2, 1, 1) -SpatialBatchNormalization(ndf*4) - nn.LeakyReLU( 0.02,true)
-- state size ; (ndf*4) x 8 x 8

   -- print(d3)
    local d4= d3  - nn.SpatialConvolution(ndf*4, ndf*8, 4, 4, 2, 2, 1, 1) -SpatialBatchNormalization(ndf*8) - nn.LeakyReLU(0.02, true)

-- state size ; (ndf*8) x 4 x 4
   local d5= d4 - nn.SpatialConvolution(ndf*8, ndf*8, 4, 4, 2, 2, 1, 1) -SpatialBatchNormalization(ndf*8) -nn.LeakyReLU(0.02,true) --nn.View(torch.LongStorage{bs,ndf*8 *2*2})

--state size ; (ndf*8) x 2x 2
    local d6= d5 -nn.View(torch.LongStorage{bs,ndf*8 *2*2}) -nn.Linear((ndf*8*2*2),1024)-nn.Dropout(0.2) -nn.LeakyReLU(0.02,true)

    --print(d4)


    local d8={d6,z2} -nn.CSubTable() -nn.Abs() -nn.Mean(2)
    netD = nn.gModule({z0,d0},{d8})
    --netD = nn.gModule({d0},{d6})

    return netD

end

function defineD_JointProb_U_Zsum_lowleak_droplast_100sp(input_nc,ndf,nz,bs)
    -- The first input is Z we upsample that to get to the image size/2 ,the seocnd input is image
    local netD= nil

    local z0=-nn.Identity()
    local z1=z0 -nn.View(bs,nz)-nn.Linear(nz,nz)-nn.Dropout(0.2)-nn.LeakyReLU( true)
    local z2=z1 -nn.Linear(nz,nz) -nn.Dropout(0.2) -nn.LeakyReLU(true)
    local z3=z2 -nn.Linear(nz,nz) -nn.Dropout(0.2) -nn.LeakyReLU(true)
    local z4=z3 -nn.Linear(nz,nz) -nn.Dropout(0.2) -nn.LeakyReLU(true)


    local d0= -nn.Identity()


-- state size : (nc*2) x 64 x 64

    local d1 = d0 - nn.SpatialConvolution(input_nc, ndf, 4, 4, 2, 2, 1, 1) - SpatialBatchNormalization(ndf) -nn.LeakyReLU(0.02,true)
--  state size ; (ndf)*32 x 32

   -- print(d1)

    local d2= d1  - nn.SpatialConvolution(ndf, ndf*2, 4, 4, 2, 2, 1, 1) -SpatialBatchNormalization (ndf*2) -nn.LeakyReLU(0.02,true)

--  state size ; (ndf*2)x16 x 16

   -- print(d2)
    local d3= d2  - nn.SpatialConvolution(ndf*2, ndf*4, 4, 4, 2, 2, 1, 1) -SpatialBatchNormalization(ndf*4) - nn.LeakyReLU( 0.02,true)
-- state size ; (ndf*4) x 8 x 8

   -- print(d3)
    local d4= d3  - nn.SpatialConvolution(ndf*4, ndf*8, 4, 4, 2, 2, 1, 1) -SpatialBatchNormalization(ndf*8) - nn.LeakyReLU(0.02, true)

-- state size ; (ndf*8) x 4 x 4
   local d5= d4 - nn.SpatialConvolution(ndf*8, ndf*8, 4, 4, 2, 2, 1, 1) -SpatialBatchNormalization(ndf*8) -nn.LeakyReLU(0.02,true) --nn.View(torch.LongStorage{bs,ndf*8 *2*2})

--state size ; (ndf*8) x 2x 2
    local d6= d5 -nn.View(torch.LongStorage{bs,ndf*8 *2*2}) -nn.Linear((ndf*8*2*2),nz)-nn.Dropout(0.2) -nn.LeakyReLU(0.02,true)

    --print(d4)


    local d8={d6,z4} -nn.CSubTable() -nn.Abs() -nn.Mean(2)
    netD = nn.gModule({z0,d0},{d8})
    --netD = nn.gModule({d0},{d6})

    return netD

end


function defineD128_JointProb_U_Zsum_lowleak_droplast(input_nc,ndf,nz,bs)
    -- The first input is Z we upsample that to get to the image size/2 ,the seocnd input is image
    local netD= nil

    local z0=-nn.Identity()
    local z1=z0 -nn.View(bs,nz)-nn.Linear(nz,1024)-nn.Dropout(0.2)-nn.LeakyReLU( true)
    local z2=z1 -nn.Linear(1024,1024) -nn.Dropout(0.2) -nn.LeakyReLU(true)
    local z3=z2 -nn.Linear(1024,1024) -nn.Dropout(0.2) -nn.LeakyReLU(true)


    local d0= -nn.Identity()



-- state size : (nc*2) x 128 x 128

    local d1 = d0 - nn.SpatialConvolution(input_nc, ndf, 4, 4, 2, 2, 1, 1) - SpatialBatchNormalization(ndf) -nn.LeakyReLU(0.02,true)

--  state size ; (ndf)*64 x 64


    local d2_= d1  - nn.SpatialConvolution(ndf, ndf*2, 4, 4, 2, 2, 1, 1) -SpatialBatchNormalization (ndf*2) -nn.LeakyReLU(0.02,true)
    local d2= d2_  -nn.SpatialConvolution(ndf*2, ndf*2, 3, 3, 1, 1, 1, 1) -SpatialBatchNormalization (ndf*2) -nn.LeakyReLU(0.02,true)

--  state size ; (ndf*2)x32 x 32

   -- print(d2)
    local d3_= d2  - nn.SpatialConvolution(ndf*2, ndf*4, 4, 4, 2, 2, 1, 1) -SpatialBatchNormalization(ndf*4) - nn.LeakyReLU( 0.02,true)
    local d3= d3_  - nn.SpatialConvolution(ndf*4, ndf*4, 3, 3, 1, 1, 1, 1) -SpatialBatchNormalization(ndf*4) - nn.LeakyReLU( 0.02,true)

-- state size ; (ndf*4) x 16 x 16

   -- print(d3)
    local d4_= d3  - nn.SpatialConvolution(ndf*4, ndf*8, 4, 4, 2, 2, 1, 1) -SpatialBatchNormalization(ndf*8) - nn.LeakyReLU(0.02, true)
    local d4= d4_  - nn.SpatialConvolution(ndf*8, ndf*8, 3, 3, 1, 1, 1, 1) -SpatialBatchNormalization(ndf*8) - nn.LeakyReLU( 0.02,true)

-- state size ; (ndf*8) x 8 x 8
   local d5_= d4 - nn.SpatialConvolution(ndf*8, ndf*8, 4, 4, 2, 2, 1, 1) -SpatialBatchNormalization(ndf*8) -nn.LeakyReLU(0.02,true) --nn.View(torch.LongStorage{bs,ndf*8 *2*2})
   local d5= d5_  - nn.SpatialConvolution(ndf*8, ndf*8, 3, 3, 1, 1, 1, 1) -SpatialBatchNormalization(ndf*8) - nn.LeakyReLU( 0.02,true)

--state size ; (ndf*8) x 4x 4
    local d6_= d5 - nn.SpatialConvolution(ndf*8, ndf*8, 4, 4, 2, 2, 1, 1) -SpatialBatchNormalization(ndf*8) -nn.LeakyReLU(0.02,true) --nn.View(torch.LongStorage{bs,ndf*8 *2*2})
 local d6= d6_  - nn.SpatialConvolution(ndf*8, ndf*8, 3, 3, 1, 1, 1, 1) -SpatialBatchNormalization(ndf*8) - nn.LeakyReLU( 0.02,true)

--state size ; (ndf*8) x 2x 2
    local d7= d6 -nn.View(torch.LongStorage{bs,ndf*8 *2*2}) -nn.Linear((ndf*8*2*2),1024)-nn.Dropout(0.2) -nn.LeakyReLU(0.02,true)

    --print(d4)


    local d8={d7,z3} -nn.CSubTable() -nn.Abs() -nn.Mean(2)
    netD = nn.gModule({z0,d0},{d8})
    --netD = nn.gModule({d0},{d6})

    return netD

end

function defineD128_shallow_JointProb_U_Zsum_lowleak_droplast(input_nc,ndf,nz,bs)
    -- The first input is Z we upsample that to get to the image size/2 ,the seocnd input is image
    local netD= nil

    local z0=-nn.Identity()
    local z1=z0 -nn.View(bs,nz)-nn.Linear(nz,1024)-nn.Dropout(0.2)-nn.LeakyReLU( true)
    local z2=z1 -nn.Linear(1024,1024) -nn.Dropout(0.2) -nn.LeakyReLU(true)
    --local z3=z2 -nn.Linear(1024,1024) -nn.Dropout(0.2) -nn.LeakyReLU(true)


    local d0= -nn.Identity()



-- state size : (nc*2) x 128 x 128

    local d1 = d0 - nn.SpatialConvolution(input_nc, ndf, 4, 4, 2, 2, 1, 1) - SpatialBatchNormalization(ndf) -nn.LeakyReLU(0.02,true)

--  state size ; (ndf)*64 x 64


    local d2_= d1  - nn.SpatialConvolution(ndf, ndf*2, 4, 4, 2, 2, 1, 1) -SpatialBatchNormalization (ndf*2) -nn.LeakyReLU(0.02,true)
    local d2= d2_  -nn.SpatialConvolution(ndf*2, ndf*2, 3, 3, 1, 1, 1, 1) -SpatialBatchNormalization (ndf*2) -nn.LeakyReLU(0.02,true)

--  state size ; (ndf*2)x32 x 32

   -- print(d2)
    local d3_= d2  - nn.SpatialConvolution(ndf*2, ndf*4, 4, 4, 2, 2, 1, 1) -SpatialBatchNormalization(ndf*4) - nn.LeakyReLU( 0.02,true)
    local d3= d3_  - nn.SpatialConvolution(ndf*4, ndf*4, 3, 3, 1, 1, 1, 1) -SpatialBatchNormalization(ndf*4) - nn.LeakyReLU( 0.02,true)

-- state size ; (ndf*4) x 16 x 16

   -- print(d3)
    local d4_= d3  - nn.SpatialConvolution(ndf*4, ndf*8, 4, 4, 2, 2, 1, 1) -SpatialBatchNormalization(ndf*8) - nn.LeakyReLU(0.02, true)
    local d4= d4_  - nn.SpatialConvolution(ndf*8, ndf*8, 3, 3, 1, 1, 1, 1) -SpatialBatchNormalization(ndf*8) - nn.LeakyReLU( 0.02,true)

-- state size ; (ndf*8) x 8 x 8
   local d5_= d4 - nn.SpatialConvolution(ndf*8, ndf*8, 4, 4, 2, 2, 1, 1) -SpatialBatchNormalization(ndf*8) -nn.LeakyReLU(0.02,true) --nn.View(torch.LongStorage{bs,ndf*8 *2*2})
   local d5= d5_  - nn.SpatialConvolution(ndf*8, ndf*8, 3, 3, 1, 1, 1, 1) -SpatialBatchNormalization(ndf*8) - nn.LeakyReLU( 0.02,true)

--state size ; (ndf*8) x 4x 4
    local d6_= d5 - nn.SpatialConvolution(ndf*8, ndf*8, 4, 4, 2, 2, 1, 1) -SpatialBatchNormalization(ndf*8) -nn.LeakyReLU(0.02,true) --nn.View(torch.LongStorage{bs,ndf*8 *2*2})
 local d6= d6_  - nn.SpatialConvolution(ndf*8, ndf*8, 3, 3, 1, 1, 1, 1) -SpatialBatchNormalization(ndf*8) - nn.LeakyReLU( 0.02,true)

--state size ; (ndf*8) x 2x 2
    local d7= d6 -nn.View(torch.LongStorage{bs,ndf*8 *2*2}) -nn.Linear((ndf*8*2*2),1024)-nn.Dropout(0.2) -nn.LeakyReLU(0.02,true)

    --print(d4)


    local d8={d7,z2} -nn.CSubTable() -nn.Abs() -nn.Mean(2)
    netD = nn.gModule({z0,d0},{d8})
    --netD = nn.gModule({d0},{d6})

    return netD

end

function defineD32_JointProb_U_Zsum_lowleak_droplast(input_nc,ndf,nz,bs)
    -- The first input is Z we upsample that to get to the image size/2 ,the seocnd input is image
    local netD= nil

    local z0=-nn.Identity()
    local z1=z0 -nn.View(bs,nz)-nn.Linear(nz,1024)-nn.Dropout(0.2)-nn.LeakyReLU( true) -- is equal to inplace true and leak =0.01
    local z2=z1 -nn.Linear(1024,1024) -nn.Dropout(0.2) -nn.LeakyReLU(true)
    --local z3=z2 -nn.Linear(1024,1024) -nn.Dropout(0.2) -nn.LeakyReLU(true)

    local d0= -nn.Identity()


-- state size : (nc*2) x 32 x 32

    local d1 = d0 - nn.SpatialConvolution(input_nc, ndf, 3, 3, 1, 1, 1, 1) - SpatialBatchNormalization(ndf) -nn.LeakyReLU(0.02,true)
--  state size ; (ndf)*32 x 32

   -- print(d1)

    local d2= d1  - nn.SpatialConvolution(ndf, ndf*2, 4, 4, 2, 2, 1, 1) -SpatialBatchNormalization (ndf*2) -nn.LeakyReLU(0.02,true)

--  state size ; (ndf*2)x16 x 16

   -- print(d2)
    local d3_= d2 - nn.SpatialConvolution(ndf*2, ndf*4, 3, 3, 1, 1, 1, 1) - SpatialBatchNormalization(ndf*4) -nn.LeakyReLU(0.02,true)
    local d3= d3_  - nn.SpatialConvolution(ndf*4, ndf*8, 4, 4, 2, 2, 1, 1) -SpatialBatchNormalization(ndf*8) - nn.LeakyReLU( 0.02,true)

-- state size ; (ndf*4) x 8 x 8

   -- print(d3)
    local d4_= d3  - nn.SpatialConvolution(ndf*8, ndf*16, 3, 3, 1, 1, 1, 1) -SpatialBatchNormalization(ndf*16) - nn.LeakyReLU(0.02, true)
    local d4= d4_  - nn.SpatialConvolution(ndf*16, ndf*16, 4, 4, 2, 2, 1, 1) -SpatialBatchNormalization(ndf*16) - nn.LeakyReLU(0.02, true)


-- state size ; (ndf*8) x 4 x 4
     local d5_= d4 - nn.SpatialConvolution(ndf*16, ndf*16, 3, 3, 1, 1, 1, 1) -SpatialBatchNormalization(ndf*16) - nn.LeakyReLU(0.02, true)
   local d5= d5_ - nn.SpatialConvolution(ndf*16, ndf*16, 4, 4, 2, 2, 1, 1) -SpatialBatchNormalization(ndf*16) -nn.LeakyReLU(0.02,true) --nn.View(torch.LongStorage{bs,ndf*8 *2*2})

--state size ; (ndf*8) x 2x 2
    local d6= d5 -nn.View(torch.LongStorage{bs,ndf*16 *2*2}) -nn.Linear((ndf*16*2*2),1024)-nn.Dropout(0.2) -nn.LeakyReLU(0.02,true)

    --print(d4)


    local d8={d6,z2} -nn.CSubTable() -nn.Abs() -nn.Mean(2)
    netD = nn.gModule({z0,d0},{d8})
    --netD = nn.gModule({d0},{d6})

    return netD

end


function defineG(nc, ngf,nz) --for the one that works the leak was 0.2 now I am going to decrease that to 0.02
    local netG = nn.Sequential()
-- input is Z, going into a convolution
netG:add(nn.SpatialFullConvolution(nz, ngf * 8, 4, 4))
netG:add(nn.SpatialBatchNormalization(ngf * 8)):add(nn.LeakyReLU(0.2, true))
-- state size: (ngf*8) x 4 x 4
--
netG:add(nn.SpatialFullConvolution(ngf * 8, ngf * 4, 4, 4, 2, 2, 1, 1))
netG:add(nn.SpatialBatchNormalization(ngf * 4)):add(nn.LeakyReLU(0.2, true))

-- state size: (ngf*4) x 8 x 8
netG:add(nn.SpatialFullConvolution(ngf * 4, ngf * 2, 4, 4, 2, 2, 1, 1))
netG:add(nn.SpatialBatchNormalization(ngf * 2)):add(nn.LeakyReLU(0.2, true))-- :add(nn.ReLU(true))
-- state size: (ngf*2) x 16 x 16
     --
netG:add(nn.SpatialFullConvolution(ngf * 2, ngf, 4, 4, 2, 2, 1, 1))
netG:add(nn.SpatialBatchNormalization(ngf)):add(nn.LeakyReLU(0.2, true))
-- state size: (ngf) x 32 x 32

netG:add(nn.SpatialFullConvolution(ngf, nc, 4, 4, 2, 2, 1, 1))
netG:add(nn.Tanh())
--]]
    return  netG
end

function defineG_BEGAN(nc,ngf,nz)
    local netG=nn.Sequential()
    -- input size vector of size nz
    netG:add(nn.View(-1,nz))--torch.LongStorage({bs,nz})

    netG:add(nn.Linear(nz,ngf*8*8))
    netG:add(nn.View(torch.LongStorage({-1,ngf,8,8})))

    --state (ngf,8,8)
    netG:add(nn.SpatialConvolution(ngf, ngf, 3, 3, 1, 1, 1, 1) )
    netG:add(nn.ELU(1,true))

    netG:add(nn.SpatialConvolution(ngf, ngf, 3, 3, 1, 1, 1, 1) )
    netG:add(nn.ELU(1,true))

    netG:add(nn.SpatialUpSamplingNearest(2))
    --state(ngf,16,16)

    netG:add(nn.SpatialConvolution(ngf, ngf, 3, 3, 1, 1, 1, 1) )
    netG:add(nn.ELU(1,true))

    netG:add(nn.SpatialConvolution(ngf, ngf, 3, 3, 1, 1, 1, 1) )
    netG:add(nn.ELU(1,true))
    netG:add(nn.SpatialUpSamplingNearest(2))

    --state(ngf,32,32)
    netG:add(nn.SpatialConvolution(ngf, ngf, 3, 3, 1, 1, 1, 1) )
    netG:add(nn.ELU(1,true))

    netG:add(nn.SpatialConvolution(ngf, ngf, 3, 3, 1, 1, 1, 1) )
    netG:add(nn.ELU(1,true))
    netG:add(nn.SpatialUpSamplingNearest(2))

    --state(ngf,64,64)
    netG:add(nn.SpatialConvolution(ngf, ngf, 3, 3, 1, 1, 1, 1) )
    netG:add(nn.ELU(1,true))

    netG:add(nn.SpatialConvolution(ngf, ngf, 3, 3, 1, 1, 1, 1) )
    netG:add(nn.ELU(1,true))

    netG:add(nn.SpatialConvolution(ngf, nc, 3, 3, 1, 1, 1, 1) ):add(nn.Tanh())--add(nn.SpatialBatchNormalization(nc))
--]]
    return netG

end


function defineG_lowleak(nc, ngf,nz) --for the one that works the leak was 0.2 now I am going to decrease that to 0.02
    local netG = nn.Sequential()
-- input is Z, going into a convolution
netG:add(nn.SpatialFullConvolution(nz, ngf * 8, 4, 4))
netG:add(nn.SpatialBatchNormalization(ngf * 8)):add(nn.LeakyReLU(0.02, true))
-- state size: (ngf*8) x 4 x 4
--
netG:add(nn.SpatialFullConvolution(ngf * 8, ngf * 4, 4, 4, 2, 2, 1, 1))
netG:add(nn.SpatialBatchNormalization(ngf * 4)):add(nn.LeakyReLU(0.02, true))

-- state size: (ngf*4) x 8 x 8
netG:add(nn.SpatialFullConvolution(ngf * 4, ngf * 2, 4, 4, 2, 2, 1, 1))
netG:add(nn.SpatialBatchNormalization(ngf * 2)):add(nn.LeakyReLU(0.02, true))-- :add(nn.ReLU(true))
-- state size: (ngf*2) x 16 x 16
     --
netG:add(nn.SpatialFullConvolution(ngf * 2, ngf, 4, 4, 2, 2, 1, 1))
netG:add(nn.SpatialBatchNormalization(ngf)):add(nn.LeakyReLU(0.02, true))
-- state size: (ngf) x 32 x 32

netG:add(nn.SpatialFullConvolution(ngf, nc, 4, 4, 2, 2, 1, 1))
netG:add(nn.Tanh())
--]]
    return  netG
end


function defineG128_noleak(nc, ngf,nz)

    local netG = nn.Sequential()
-- input is Z, going into a convolution
netG:add(nn.SpatialFullConvolution(nz, ngf * 8, 4, 4))
netG:add(nn.SpatialBatchNormalization(ngf * 8)):add(nn.ReLU( true))
-- state size: (ngf*8) x 4 x 4
--
netG:add(nn.SpatialFullConvolution(ngf * 8, ngf * 4, 4, 4, 2, 2, 1, 1))
netG:add(nn.SpatialBatchNormalization(ngf * 4)):add(nn.ReLU( true))

netG:add(nn.SpatialConvolution(ngf*4, 4*ngf, 3, 3, 1, 1, 1, 1) )
netG:add(nn.SpatialBatchNormalization(ngf * 4)):add(nn.ReLU( true))

-- state size: (ngf*4) x 8 x 8
netG:add(nn.SpatialFullConvolution(ngf * 4, ngf * 4, 4, 4, 2, 2, 1, 1))
netG:add(nn.SpatialBatchNormalization(ngf * 4)):add(nn.ReLU(true))-- :add(nn.ReLU(true))

netG:add(nn.SpatialConvolution(ngf*4, 4*ngf, 3, 3, 1, 1, 1, 1) )
netG:add(nn.SpatialBatchNormalization(ngf * 4)):add(nn.ReLU( true))

-- state size: (ngf*4) x 16 x 16
     --
netG:add(nn.SpatialFullConvolution(ngf * 4, ngf*2, 4, 4, 2, 2, 1, 1))
netG:add(nn.SpatialBatchNormalization(ngf*2)):add(nn.ReLU(true))

netG:add(nn.SpatialConvolution(ngf*2, 2*ngf, 3, 3, 1, 1, 1, 1) )
netG:add(nn.SpatialBatchNormalization(ngf * 2)):add(nn.ReLU( true))

-- state size: (ngf*2) x 32 x 32
netG:add(nn.SpatialFullConvolution(ngf * 2, ngf*2, 4, 4, 2, 2, 1, 1))
netG:add(nn.SpatialBatchNormalization(ngf*2)):add(nn.ReLU(true))

netG:add(nn.SpatialConvolution(ngf*2, ngf, 3, 3, 1, 1, 1, 1) )
netG:add(nn.SpatialBatchNormalization(ngf)):add(nn.ReLU( true))
-- state size: (ngf) x 64 x 64

netG:add(nn.SpatialFullConvolution(ngf, nc, 4, 4, 2, 2, 1, 1))
netG:add(nn.Tanh())
-- state size nc x 128x128
--]]
    return  netG
end

function defineG128_compare_lsgan(nc, ngf,nz)
    local netG = nn.Sequential()

-- input is Z, going into a convolution
netG:add(SpatialFullConvolution(nz, ngf * 16, 4, 4))
netG:add(SpatialBatchNormalization(ngf * 16)):add(nn.ReLU(true))
-- state size: (ngf*8) x 4 x 4
netG:add(SpatialFullConvolution(ngf * 16, ngf * 8, 4, 4, 2, 2, 1, 1))
netG:add(SpatialBatchNormalization(ngf * 8)):add(nn.ReLU(true))
-- state size: (ngf*4) x 8 x 8
netG:add(SpatialFullConvolution(ngf * 8, ngf * 4, 4, 4, 2, 2, 1, 1))
netG:add(SpatialBatchNormalization(ngf * 4)):add(nn.ReLU(true))
-- state size: (ngf*2) x 16 x 16
netG:add(SpatialFullConvolution(ngf * 4, 2*ngf, 4, 4, 2, 2, 1, 1))
netG:add(SpatialBatchNormalization(2*ngf)):add(nn.ReLU(true))
-- state size: (ngf) x 32 x 32
netG:add(SpatialFullConvolution(ngf * 2, ngf, 4, 4, 2, 2, 1, 1))
netG:add(SpatialBatchNormalization(ngf)):add(nn.ReLU(true))

netG:add(SpatialFullConvolution(ngf, nc, 4, 4, 2, 2, 1, 1))
netG:add(nn.Tanh())

    return netG
end

function defineG128_noleak_shallow(nc, ngf,nz)

    local netG = nn.Sequential()
-- input is Z, going into a convolution
netG:add(nn.SpatialFullConvolution(nz, ngf * 16, 4, 4))
netG:add(nn.SpatialBatchNormalization(ngf * 16)):add(nn.ReLU( true))
-- state size: (ngf*8) x 4 x 4
--
netG:add(nn.SpatialFullConvolution(ngf * 16, ngf * 8, 4, 4, 2, 2, 1, 1))
netG:add(nn.SpatialBatchNormalization(ngf * 8)):add(nn.ReLU( true))

--netG:add(nn.SpatialConvolution(ngf*4, 4*ngf, 3, 3, 1, 1, 1, 1) )
--netG:add(nn.SpatialBatchNormalization(ngf * 4)):add(nn.ReLU( true))

-- state size: (ngf*4) x 8 x 8
netG:add(nn.SpatialFullConvolution(ngf * 8, ngf * 4, 4, 4, 2, 2, 1, 1))
netG:add(nn.SpatialBatchNormalization(ngf * 4)):add(nn.ReLU(true))-- :add(nn.ReLU(true))

--netG:add(nn.SpatialConvolution(ngf*4, 4*ngf, 3, 3, 1, 1, 1, 1) )
--netG:add(nn.SpatialBatchNormalization(ngf * 4)):add(nn.ReLU( true))

-- state size: (ngf*4) x 16 x 16
     --
netG:add(nn.SpatialFullConvolution(ngf * 4, ngf*2, 4, 4, 2, 2, 1, 1))
netG:add(nn.SpatialBatchNormalization(ngf*2)):add(nn.ReLU(true))

--netG:add(nn.SpatialConvolution(ngf*2, 2*ngf, 3, 3, 1, 1, 1, 1) )
--netG:add(nn.SpatialBatchNormalization(ngf * 2)):add(nn.ReLU( true))

-- state size: (ngf*2) x 32 x 32
netG:add(nn.SpatialFullConvolution(ngf * 2, ngf*2, 4, 4, 2, 2, 1, 1))
netG:add(nn.SpatialBatchNormalization(ngf*2)):add(nn.ReLU(true))

netG:add(nn.SpatialConvolution(ngf*2, ngf, 3, 3, 1, 1, 1, 1) )
netG:add(nn.SpatialBatchNormalization(ngf)):add(nn.ReLU( true))
-- state size: (ngf) x 64 x 64

netG:add(nn.SpatialFullConvolution(ngf, nc, 4, 4, 2, 2, 1, 1))
netG:add(nn.Tanh())
-- state size nc x 128x128
--]]
    return  netG
end

function defineG_noleak(nc, ngf,nz)
    local netG = nn.Sequential()
-- input is Z, going into a convolution
netG:add(nn.SpatialFullConvolution(nz, ngf * 8, 4, 4))
netG:add(nn.SpatialBatchNormalization(ngf * 8)):add(nn.ReLU( true))

-- state size: (ngf*8) x 4 x 4
--
netG:add(nn.SpatialFullConvolution(ngf * 8, ngf * 4, 4, 4, 2, 2, 1, 1))
netG:add(nn.SpatialBatchNormalization(ngf * 4)):add(nn.ReLU( true))

-- state size: (ngf*4) x 8 x 8
netG:add(nn.SpatialFullConvolution(ngf * 4, ngf * 2, 4, 4, 2, 2, 1, 1))
netG:add(nn.SpatialBatchNormalization(ngf * 2)):add(nn.ReLU(true))-- :add(nn.ReLU(true))
-- state size: (ngf*2) x 16 x 16
     --
netG:add(nn.SpatialFullConvolution(ngf * 2, ngf, 4, 4, 2, 2, 1, 1))
netG:add(nn.SpatialBatchNormalization(ngf)):add(nn.ReLU(true))
-- state size: (ngf) x 32 x 32

netG:add(nn.SpatialFullConvolution(ngf, nc, 4, 4, 2, 2, 1, 1))
netG:add(nn.Tanh())
--]]
    return  netG
end


function defineG32_noleak(nc, ngf,nz)
    local netG = nn.Sequential()
-- input is Z, going into a convolution
-----netG:add(nn.View(-1,nz)):add(nn.Linear(nz,ngf*8*4*4)):add(nn.View(-1,ngf*8,4,4)):add(nn.ReLU(true))
netG:add(nn.SpatialFullConvolution(nz, ngf * 8, 4, 4))
netG:add(nn.SpatialBatchNormalization(ngf * 8)):add(nn.ReLU( true))

-- state size: (ngf*8) x 4 x 4
--
netG:add(nn.SpatialFullConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1))
netG:add(nn.SpatialBatchNormalization(ngf * 8)):add(nn.ReLU( true))

netG:add(SpatialConvolution(ngf * 8, ngf*8 , 3, 3, 1 , 1, 1, 1))
netG:add(SpatialBatchNormalization(ngf *8)):add(nn.ReLU( true))


-- state size: (ngf*4) x 8 x 8
netG:add(nn.SpatialFullConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1))
netG:add(nn.SpatialBatchNormalization(ngf * 8)):add(nn.ReLU(true))-- :add(nn.ReLU(true))

netG:add(SpatialConvolution(ngf * 8, ngf*4 , 3, 3, 1 , 1, 1, 1))
netG:add(SpatialBatchNormalization(ngf *4)):add(nn.ReLU( true))

-- state size: (ngf*2) x 16 x 16

netG:add(SpatialConvolution(ngf * 4, ngf , 3, 3, 1 , 1, 1, 1))
netG:add(SpatialBatchNormalization(ngf )):add(nn.ReLU( true))

netG:add(nn.SpatialFullConvolution(ngf, nc, 4, 4, 2, 2, 1, 1))
netG:add(nn.Tanh())
-- state size: (ngf) x 32 x 32
    return  netG
end

function defineG_dense(nc, ngf,nz,bs)
    local netG = nn.Sequential()

netG:add(nn.View(bs,nz)):add(nn.Linear(nz,1024)):add(nn.ReLU(true)) --:add(nn.Dropout(0.2))
netG:add(nn.Linear(1024,1024)):add(nn.LeakyReLU(0.2,true)):add(nn.View(bs,1024,1,1)) --:add(nn.Dropout(0.2))
-- input is Z, going into a convolution
netG:add(nn.SpatialFullConvolution(1024, ngf * 8, 4, 4))
netG:add(nn.SpatialBatchNormalization(ngf * 8)):add(nn.LeakyReLU(0.2, true))
-- state size: (ngf*8) x 4 x 4
--
netG:add(nn.SpatialFullConvolution(ngf * 8, ngf * 4, 4, 4, 2, 2, 1, 1))
netG:add(nn.SpatialBatchNormalization(ngf * 4)):add(nn.LeakyReLU(0.2, true))

-- state size: (ngf*4) x 8 x 8
netG:add(nn.SpatialFullConvolution(ngf * 4, ngf * 2, 4, 4, 2, 2, 1, 1))
netG:add(nn.SpatialBatchNormalization(ngf * 2)):add(nn.LeakyReLU(0.2, true))-- :add(nn.ReLU(true))
-- state size: (ngf*2) x 16 x 16
     --
netG:add(nn.SpatialFullConvolution(ngf * 2, ngf, 4, 4, 2, 2, 1, 1))
netG:add(nn.SpatialBatchNormalization(ngf)):add(nn.LeakyReLU(0.2, true))
-- state size: (ngf) x 32 x 32

netG:add(nn.SpatialFullConvolution(ngf, nc, 4, 4, 2, 2, 1, 1))
netG:add(nn.Tanh())
--]]
    return  netG
end

function defineQ(nc,nqf,nz)
local netQ = nn.Sequential()
    -- input is (nc) x 64 x 64
netQ:add(SpatialConvolution(nc, nqf, 4, 4, 2, 2, 1, 1))
netQ:add(nn.LeakyReLU(0.2, true))
-- state size: (nqf) x 32 x 32
netQ:add(SpatialConvolution(nqf, nqf * 2, 4, 4, 2, 2, 1, 1))
netQ:add(SpatialBatchNormalization(nqf * 2)):add(nn.LeakyReLU(0.2, true))--
-- state size: (nqf*2) x 16 x 16
netQ:add(SpatialConvolution(nqf * 2, nqf * 4, 4, 4, 2, 2, 1, 1))
netQ:add(SpatialBatchNormalization(nqf * 4)):add(nn.LeakyReLU(0.2, true))--
-- state size: (nqf*4) x 8 x 8
netQ:add(SpatialConvolution(nqf * 4, nqf * 8, 4, 4, 2, 2, 1, 1))
netQ:add(SpatialBatchNormalization(nqf * 8)):add(nn.LeakyReLU(0.2, true))--
-- state size: (nqf*8) x 4 x 4
netQ:add(SpatialConvolution(nqf * 8, nz, 4, 4))
    return netQ
end

function defineQ_BEGAN(nc,nqf,nz,bs)
local netQ = nn.Sequential()
    -- input is (nc) x 64 x 64
netQ:add(SpatialConvolution(nc, nqf, 3, 3, 1, 1, 1, 1))
netQ:add(nn.ELU(1, true))

netQ:add(SpatialConvolution(nqf, nqf, 3, 3, 1, 1, 1, 1))
netQ:add(nn.ELU(1, true))

netQ:add(SpatialConvolution(nqf, nqf, 3, 3, 1, 1, 1, 1))
netQ:add(nn.ELU(1, true))

netQ:add(SpatialConvolution(nqf, 2*nqf, 1, 1))
netQ:add(nn.SpatialAveragePooling(2,2,2,2))
--state(2 x nqf,32,32)
netQ:add(SpatialConvolution(2*nqf, 2*nqf, 3, 3, 1, 1, 1, 1))
netQ:add(nn.ELU(1, true))

netQ:add(SpatialConvolution(2*nqf, 2*nqf, 3, 3, 1, 1, 1, 1))
netQ:add(nn.ELU(1, true))

netQ:add(SpatialConvolution(2*nqf, 3*nqf, 1, 1))
netQ:add(nn.SpatialAveragePooling(2,2,2,2))
--state(3 x nqf,16,16)
netQ:add(SpatialConvolution(3*nqf, 3*nqf, 3, 3, 1, 1, 1, 1))
netQ:add(nn.ELU(1, true))

netQ:add(SpatialConvolution(3*nqf, 3*nqf, 3, 3, 1, 1, 1, 1))
netQ:add(nn.ELU(1, true))

netQ:add(SpatialConvolution(3*nqf, 4*nqf, 1, 1))
netQ:add(nn.SpatialAveragePooling(2,2,2,2))
--state(4 x nqf,8,8)

netQ:add(SpatialConvolution(4*nqf, 4*nqf, 3, 3, 1, 1, 1, 1))
netQ:add(nn.ELU(1, true))

netQ:add(SpatialConvolution(4*nqf, 4*nqf, 3, 3, 1, 1, 1, 1))
netQ:add(nn.ELU(1, true))

--netQ:add(SpatialConvolution(4*nqf, 5*nqf, 1, 1))
--netQ:add(nn.SpatialAveragePooling(2,2,2,2))
--state(5 x nqf,4,4)
netQ:add(SpatialConvolution(nqf*4,nz,8,8)):add(SpatialBatchNormalization(nz)):add(nn.Tanh())--add(nn.ELU(1,true))
--netQ:add(nn.View(bs,-1)):add(nn.Linear(nqf*4*8*8,nz)):add(nn.Tanh())


return netQ

end
--]]


function defineQ_BNLL(nc,nqf,nz)
local netQ = nn.Sequential()
    -- input is (nc) x 64 x 64
netQ:add(SpatialConvolution(nc, nqf, 4, 4, 2, 2, 1, 1))
netQ:add(nn.LeakyReLU(0.2, true))
-- state size: (nqf) x 32 x 32
netQ:add(SpatialConvolution(nqf, nqf * 2, 4, 4, 2, 2, 1, 1))
netQ:add(SpatialBatchNormalization(nqf * 2)):add(nn.LeakyReLU(0.2, true))--
-- state size: (nqf*2) x 16 x 16
netQ:add(SpatialConvolution(nqf * 2, nqf * 4, 4, 4, 2, 2, 1, 1))
netQ:add(SpatialBatchNormalization(nqf * 4)):add(nn.LeakyReLU(0.2, true))--
-- state size: (nqf*4) x 8 x 8
netQ:add(SpatialConvolution(nqf * 4, nqf * 8, 4, 4, 2, 2, 1, 1))
netQ:add(SpatialBatchNormalization(nqf * 8)):add(nn.LeakyReLU(0.2, true))--
-- state size: (nqf*8) x 4 x 4
netQ:add(SpatialConvolution(nqf * 8, nz, 4, 4)):add(SpatialBatchNormalization(nz)): add(nn.Tanh())--:add(nn.AddConstant(1,true)):add(nn.LeakyReLU(0.2, true)):add(nn.AddConstant(-1,true))--shift relu
    return netQ
end

function defineQ_noleak(nc,nqf,nz)
local netQ = nn.Sequential()
    -- input is (nc) x 64 x 64
netQ:add(SpatialConvolution(nc, nqf, 4, 4, 2, 2, 1, 1))
netQ:add(nn.ReLU(true))
-- state size: (nqf) x 32 x 32
netQ:add(SpatialConvolution(nqf, nqf * 2, 4, 4, 2, 2, 1, 1))
netQ:add(SpatialBatchNormalization(nqf * 2)):add(nn.ReLU( true))--
-- state size: (nqf*2) x 16 x 16
netQ:add(SpatialConvolution(nqf * 2, nqf * 4, 4, 4, 2, 2, 1, 1))
netQ:add(SpatialBatchNormalization(nqf * 4)):add(nn.ReLU( true))--
-- state size: (nqf*4) x 8 x 8
netQ:add(SpatialConvolution(nqf * 4, nqf * 8, 4, 4, 2, 2, 1, 1))
netQ:add(SpatialBatchNormalization(nqf * 8)):add(nn.ReLU(true))--
-- state size: (nqf*8) x 4 x 4
netQ:add(SpatialConvolution(nqf * 8, nz, 4, 4)):add(SpatialBatchNormalization(nz)):add(nn.Tanh())
    return netQ                               ---
end


function defineQ32_noleak(nc,nqf,nz)
local netQ = nn.Sequential()
    -- input is (nc) x 32 x 32
netQ:add(SpatialConvolution(nc, nqf*2, 4, 4, 2, 2, 1, 1))
netQ:add(nn.ReLU(true))
-- state size: (nqf*2) x 32 x 32
netQ:add(SpatialConvolution(nqf*2, nqf * 4, 3, 3, 1, 1, 1, 1))
netQ:add(SpatialBatchNormalization(nqf * 4)):add(nn.ReLU( true))--
-- state size: (nqf*2) x 16 x 16
netQ:add(SpatialConvolution(nqf * 4, nqf * 8, 4, 4, 2, 2, 1, 1))
netQ:add(SpatialBatchNormalization(nqf * 8)):add(nn.ReLU( true))--

netQ:add(SpatialConvolution(nqf*8, nqf * 8, 3, 3, 1, 1, 1, 1))
netQ:add(SpatialBatchNormalization(nqf * 8)):add(nn.ReLU( true))--

-- state size: (nqf*4) x 8 x 8
netQ:add(SpatialConvolution(nqf * 8, nqf * 16, 4, 4, 2, 2, 1, 1))
netQ:add(SpatialBatchNormalization(nqf * 16)):add(nn.ReLU(true))--
-- state size: (nqf*8) x 4 x 4
netQ:add(SpatialConvolution(nqf * 16, nz, 4, 4)):add(SpatialBatchNormalization(nz)):add(nn.Tanh())
    return netQ                               ---
end
function defineQ32_BEGAN(nc,nqf,nz,bs)
local netQ = nn.Sequential()
    -- input is (nc) x 32 x 32
netQ:add(SpatialConvolution(nc, nqf, 3, 3, 1, 1, 1, 1))
netQ:add(nn.ELU(1, true))

netQ:add(SpatialConvolution(nqf, nqf, 3, 3, 1, 1, 1, 1))
netQ:add(nn.ELU(1, true))

netQ:add(SpatialConvolution(nqf, nqf, 3, 3, 1, 1, 1, 1))
netQ:add(nn.ELU(1, true))

netQ:add(SpatialConvolution(nqf, 2*nqf, 1, 1))
netQ:add(nn.SpatialAveragePooling(2,2,2,2))
--state(2 x nqf,16,16)
netQ:add(SpatialConvolution(2*nqf, 2*nqf, 3, 3, 1, 1, 1, 1))
netQ:add(nn.ELU(1, true))

netQ:add(SpatialConvolution(2*nqf, 2*nqf, 3, 3, 1, 1, 1, 1))
netQ:add(nn.ELU(1, true))

netQ:add(SpatialConvolution(2*nqf, 4*nqf, 1, 1))
netQ:add(nn.SpatialAveragePooling(2,2,2,2))
--[[
--state(3 x nqf,16,16)
netQ:add(SpatialConvolution(3*nqf, 3*nqf, 3, 3, 1, 1, 1, 1))
netQ:add(nn.ELU(1, true))

netQ:add(SpatialConvolution(3*nqf, 3*nqf, 3, 3, 1, 1, 1, 1))
netQ:add(nn.ELU(1, true))

netQ:add(SpatialConvolution(3*nqf, 4*nqf, 1, 1))
netQ:add(nn.SpatialAveragePooling(2,2,2,2))
--]]
--state(4 x nqf,8,8)

netQ:add(SpatialConvolution(4*nqf, 4*nqf, 3, 3, 1, 1, 1, 1))
netQ:add(nn.ELU(1, true))

netQ:add(SpatialConvolution(4*nqf, 4*nqf, 3, 3, 1, 1, 1, 1))
netQ:add(nn.ELU(1, true))

--netQ:add(SpatialConvolution(4*nqf, 5*nqf, 1, 1))
--netQ:add(nn.SpatialAveragePooling(2,2,2,2))
--state(5 x nqf,4,4)
netQ:add(SpatialConvolution(nqf*4,nz,8,8)):add(SpatialBatchNormalization(nz))--:add(nn.Tanh())--add(nn.ELU(1,true))
--netQ:add(nn.View(bs,-1)):add(nn.Linear(nqf*4*8*8,nz)):add(nn.Tanh())


return netQ

end

function defineQ32_noleak_classification(nc,nqf,nz)
local netQ = nn.Sequential()
    -- input is (nc) x 32 x 32
netQ:add(SpatialConvolution(nc,nqf*2,3,3,1,1,1,1)):add(SpatialBatchNormalization(nqf * 2)):add(nn.ReLU(true))
--netQ:add(SpatialConvolution(nqf*2, nqf*4, 4, 4, 2, 2, 1, 1))
--netQ:add(nn.ReLU(true))
-- state size: (nqf*2) x 32 x 32
netQ:add(SpatialConvolution(nqf*2, nqf * 4, 4, 4, 2, 2, 1, 1))
netQ:add(SpatialBatchNormalization(nqf * 4)):add(nn.ReLU( true))--
-- state size: (nqf*2) x 16 x 16
netQ:add(SpatialConvolution(nqf*4, nqf * 8, 3, 3, 1, 1, 1, 1))
netQ:add(SpatialBatchNormalization(nqf * 8)):add(nn.ReLU( true))

netQ:add(SpatialConvolution(nqf * 8, nqf * 16, 4, 4, 2, 2, 1, 1))
netQ:add(SpatialBatchNormalization(nqf * 16)):add(nn.ReLU( true))--

-- state size: (nqf*4) x 8 x 8
netQ:add(SpatialConvolution(nqf*16, nqf * 16, 3, 3, 1, 1, 1, 1))
netQ:add(SpatialBatchNormalization(nqf *16)):add(nn.ReLU( true))

netQ:add(SpatialConvolution(nqf * 16, nqf * 16, 4, 4, 2, 2, 1, 1))
netQ:add(SpatialBatchNormalization(nqf * 16)):add(nn.ReLU(true))--
-- state size: (nqf*16) x 4 x 4
netQ:add(SpatialConvolution(nqf*16, nqf * 16, 3, 3, 1, 1, 1, 1))
netQ:add(SpatialBatchNormalization(nqf *16)):add(nn.ReLU( true))

netQ:add(SpatialConvolution(nqf * 16, nqf * 16, 4, 4, 2, 2, 1, 1))
netQ:add(SpatialBatchNormalization(nqf * 16)):add(nn.ReLU(true))--
-- state size: (nqf*16) x 2 x 2
netQ:add(SpatialConvolution(nqf * 16, nz, 2, 2)):add(SpatialBatchNormalization(nz)):add(nn.Tanh())
    return netQ                               ---
end

function defineQ128_noleak(nc,nqf,nz)
local netQ = nn.Sequential()
    -- input is (nc) x 128 x 128
netQ:add(SpatialConvolution(nc, nqf, 4, 4, 2, 2, 1, 1))
netQ:add(nn.ReLU(true))
-- state size: (nqf) x 64 x 64
netQ:add(SpatialConvolution(nqf, nqf * 2, 4, 4, 2, 2, 1, 1))
netQ:add(SpatialBatchNormalization(nqf * 2)):add(nn.ReLU( true))--

netQ:add(SpatialConvolution(nqf * 2, nqf * 2, 3, 3, 1 , 1, 1, 1))
netQ:add(SpatialBatchNormalization(nqf * 2)):add(nn.ReLU( true))
-- state size: (nqf*2) x 32 x 32

netQ:add(SpatialConvolution(nqf * 2, nqf * 4, 4, 4, 2, 2, 1, 1))
netQ:add(SpatialBatchNormalization(nqf * 4)):add(nn.ReLU( true))--

netQ:add(SpatialConvolution(nqf * 4, nqf * 4, 3, 3, 1 , 1, 1, 1))
netQ:add(SpatialBatchNormalization(nqf * 4)):add(nn.ReLU( true))


-- state size: (nqf*4) x 16 x 16

netQ:add(SpatialConvolution(nqf * 4, nqf * 8, 4, 4, 2, 2, 1, 1))
netQ:add(SpatialBatchNormalization(nqf * 8)):add(nn.ReLU(true))--

netQ:add(SpatialConvolution(nqf * 8, nqf * 8, 3, 3, 1 , 1, 1, 1))
netQ:add(SpatialBatchNormalization(nqf * 8)):add(nn.ReLU( true))

-- state size: (nqf*8) x 8 x 8

netQ:add(SpatialConvolution(nqf * 8, nqf * 16, 4, 4, 2, 2, 1, 1))
netQ:add(SpatialBatchNormalization(nqf * 16)):add(nn.ReLU(true))--

netQ:add(SpatialConvolution(nqf * 16, nqf * 16, 3, 3, 1 , 1, 1, 1))
netQ:add(SpatialBatchNormalization(nqf * 16)):add(nn.ReLU( true))


-- state size: (nqf*16) x 4 x 4

netQ:add(SpatialConvolution(nqf * 16, nqf * 16, 4, 4, 2, 2, 1, 1))
netQ:add(SpatialBatchNormalization(nqf * 16)):add(nn.ReLU(true))--

netQ:add(SpatialConvolution(nqf * 16, nqf * 16, 3, 3, 1 , 1, 1, 1))
netQ:add(SpatialBatchNormalization(nqf * 16)):add(nn.ReLU( true))

--state size: (nqf*16) x 2 x 2
netQ:add(SpatialConvolution(nqf * 16, nz, 2, 2)):add(SpatialBatchNormalization(nz)):add(nn.Tanh())
    return netQ                               ---
end

function defineQ128_compare_lsgan(nc,ndf,nz)

    local netQ = nn.Sequential()

--local netD = nn.Sequential()

-- input is (nc) x 128 x 128
netQ:add(SpatialConvolution(nc, ndf, 4, 4, 2, 2, 1, 1))
netQ:add(nn.LeakyReLU(0.2, true))
-- state size: (ndf) x 64 x 64
netQ:add(SpatialConvolution(ndf, ndf * 2, 4, 4, 2, 2, 1, 1))
netQ:add(SpatialBatchNormalization(ndf * 2)):add(nn.LeakyReLU(0.2, true))
-- state size: (ndfx2) x 32 x 32
netQ:add(SpatialConvolution(ndf*2, ndf * 4, 4, 4, 2, 2, 1, 1))
netQ:add(SpatialBatchNormalization(ndf * 4)):add(nn.LeakyReLU(0.2, true))
-- state size: (ndf*4) x 16 x 16
netQ:add(SpatialConvolution(ndf * 4, ndf * 8, 4, 4, 2, 2, 1, 1))
netQ:add(SpatialBatchNormalization(ndf * 8)):add(nn.LeakyReLU(0.2, true))
-- state size: (ndf*8) x 8 x 8
netQ:add(SpatialConvolution(ndf * 8, ndf * 8, 4, 4, 2, 2, 1, 1))
netQ:add(SpatialBatchNormalization(ndf * 8)):add(nn.LeakyReLU(0.2, true))
-- state size: (ndf*8) x 4 x 4
netQ:add(SpatialConvolution(ndf * 8, nz, 4, 4)):add(SpatialBatchNormalization(nz)):add(nn.Tanh())

    return netQ
end

function defineQ128_noleak_shallow(nc,nqf,nz)
local netQ = nn.Sequential()
    -- input is (nc) x 128 x 128
netQ:add(SpatialConvolution(nc, nqf, 4, 4, 2, 2, 1, 1))
netQ:add(nn.ReLU(true))
-- state size: (nqf) x 64 x 64
netQ:add(SpatialConvolution(nqf, nqf * 2, 4, 4, 2, 2, 1, 1))
netQ:add(SpatialBatchNormalization(nqf * 2)):add(nn.ReLU( true))--

netQ:add(SpatialConvolution(nqf * 2, nqf * 2, 3, 3, 1 , 1, 1, 1))
netQ:add(SpatialBatchNormalization(nqf * 2)):add(nn.ReLU( true))
-- state size: (nqf*2) x 32 x 32

netQ:add(SpatialConvolution(nqf * 2, nqf * 4, 4, 4, 2, 2, 1, 1))
netQ:add(SpatialBatchNormalization(nqf * 4)):add(nn.ReLU( true))--

--netQ:add(SpatialConvolution(nqf * 4, nqf * 4, 3, 3, 1 , 1, 1, 1))
--netQ:add(SpatialBatchNormalization(nqf * 4)):add(nn.ReLU( true))


-- state size: (nqf*4) x 16 x 16

netQ:add(SpatialConvolution(nqf * 4, nqf * 8, 4, 4, 2, 2, 1, 1))
netQ:add(SpatialBatchNormalization(nqf * 8)):add(nn.ReLU(true))--

netQ:add(SpatialConvolution(nqf * 8, nqf * 8, 3, 3, 1 , 1, 1, 1))
netQ:add(SpatialBatchNormalization(nqf * 8)):add(nn.ReLU( true))

-- state size: (nqf*8) x 8 x 8

netQ:add(SpatialConvolution(nqf * 8, nqf * 16, 4, 4, 2, 2, 1, 1))
netQ:add(SpatialBatchNormalization(nqf * 16)):add(nn.ReLU(true))--

netQ:add(SpatialConvolution(nqf * 16, nqf * 16, 3, 3, 1 , 1, 1, 1))
netQ:add(SpatialBatchNormalization(nqf * 16)):add(nn.ReLU( true))


-- state size: (nqf*16) x 4 x 4

netQ:add(SpatialConvolution(nqf * 16, nqf * 16, 4, 4, 2, 2, 1, 1))
netQ:add(SpatialBatchNormalization(nqf * 16)):add(nn.ReLU(true))--

--netQ:add(SpatialConvolution(nqf * 16, nqf * 16, 3, 3, 1 , 1, 1, 1))
--netQ:add(SpatialBatchNormalization(nqf * 16)):add(nn.ReLU( true))

--state size: (nqf*16) x 2 x 2
netQ:add(SpatialConvolution(nqf * 16, nz, 2, 2)):add(SpatialBatchNormalization(nz)):add(nn.Tanh())
    return netQ                               ---
end

function defineQ_noleak_BEGAN(nc,nqf,nz)
local netQ = nn.Sequential()
    -- input is (nc) x 64 x 64
netQ:add(SpatialConvolution(nc, nqf, 3, 3, 1, 1, 1, 1))
netQ:add(SpatialBatchNormalization(nqf)):add(nn.ReLU( true))
netQ:add(SpatialConvolution(nqf, nqf, 4, 4, 2, 2, 1, 1))
netQ:add(nn.ReLU(true))
-- state size: (nqf) x 32 x 32
netQ:add(SpatialConvolution(nqf, 2*nqf, 3, 3, 1, 1, 1, 1))
netQ:add(SpatialBatchNormalization(2*nqf)):add(nn.ReLU( true))
netQ:add(SpatialConvolution(2*nqf, nqf * 2, 4, 4, 2, 2, 1, 1))
netQ:add(SpatialBatchNormalization(nqf * 2)):add(nn.ReLU( true))--
-- state size: (nqf*2) x 16 x 16
netQ:add(SpatialConvolution(nqf*2, 4*nqf, 3, 3, 1, 1, 1, 1))
netQ:add(SpatialBatchNormalization(4*nqf)):add(nn.ReLU( true))
netQ:add(SpatialConvolution(nqf * 4, nqf * 4, 4, 4, 2, 2, 1, 1))
netQ:add(SpatialBatchNormalization(nqf * 4)):add(nn.ReLU( true))--
-- state size: (nqf*4) x 8 x 8
netQ:add(SpatialConvolution(nqf*4, 8*nqf, 3, 3, 1, 1, 1, 1))
netQ:add(SpatialBatchNormalization(8*nqf)):add(nn.ReLU( true))
netQ:add(SpatialConvolution(nqf*8, 8*nqf, 3, 3, 1, 1, 1, 1))
netQ:add(SpatialBatchNormalization(8*nqf)):add(nn.ReLU( true))
netQ:add(SpatialConvolution(nqf * 8, nqf * 8, 4, 4, 2, 2, 1, 1))
netQ:add(SpatialBatchNormalization(nqf * 8)):add(nn.ReLU(true))--
-- state size: (nqf*8) x 4 x 4
netQ:add(SpatialConvolution(nqf * 8, nz, 4, 4)):add(SpatialBatchNormalization(nz)):add(nn.Tanh())
    return netQ                               ---
end



function defineQ_lowleak(nc,nqf,nz)
local netQ = nn.Sequential()
    -- input is (nc) x 64 x 64
netQ:add(SpatialConvolution(nc, nqf, 4, 4, 2, 2, 1, 1))
netQ:add(nn.LeakyReLU(0.02, true))
-- state size: (nqf) x 32 x 32
netQ:add(SpatialConvolution(nqf, nqf * 2, 4, 4, 2, 2, 1, 1))
netQ:add(SpatialBatchNormalization(nqf * 2)):add(nn.LeakyReLU(0.02, true))--
-- state size: (nqf*2) x 16 x 16
netQ:add(SpatialConvolution(nqf * 2, nqf * 4, 4, 4, 2, 2, 1, 1))
netQ:add(SpatialBatchNormalization(nqf * 4)):add(nn.LeakyReLU(0.02, true))--
-- state size: (nqf*4) x 8 x 8
netQ:add(SpatialConvolution(nqf * 4, nqf * 8, 4, 4, 2, 2, 1, 1))
netQ:add(SpatialBatchNormalization(nqf * 8)):add(nn.LeakyReLU(0.02, true))--
-- state size: (nqf*8) x 4 x 4
netQ:add(SpatialConvolution(nqf * 8, nz, 4, 4)) :add(SpatialBatchNormalization(nz)):add(nn.Tanh())
    return netQ                                ----
end

--New Models


function defineD_JointProb_U_Zconcat(input_nc,ndf,nz,bs)
    -- The first input is Z we upsample that to get to the image size/2 ,the seocnd input is image
    local netD= nil

    local z0=-nn.Identity()
    local z1=z0 -nn.View(bs,nz)-nn.Linear(nz,1024)-nn.Dropout(0.2)-nn.LeakyReLU(0.02, true)
    local z2=z1 -nn.Linear(1024,1024) -nn.Dropout(0.2) -nn.LeakyReLU(0.02,true) -nn.View(bs,1024,1,1)


    local d0= -nn.Identity()


-- state size : (nc*2) x 64 x 64
    local d0_= d0 -nn.SpatialConvolution(input_nc,ndf,3,3,1,1,1,1) -SpatialBatchNormalization(ndf) -nn.LeakyReLU(0.02,true)

    local d1 = d0_ - nn.SpatialConvolution(ndf, 2*ndf, 4, 4, 2, 2, 1, 1) - SpatialBatchNormalization(2*ndf) -nn.LeakyReLU(0.02,true)
--  state size ; (ndf)*32 x 32

   -- print(d1)

    local d2= d1  - nn.SpatialConvolution(2*ndf, ndf*2, 4, 4, 2, 2, 1, 1) -SpatialBatchNormalization (ndf*2) -nn.LeakyReLU(0.02,true)

--  state size ; (ndf*2)x16 x 16

   -- print(d2)
    local d3= d2  - nn.SpatialConvolution(ndf*2, ndf*4, 4, 4, 2, 2, 1, 1) -SpatialBatchNormalization(ndf*4) - nn.LeakyReLU(0.02, true)
-- state size ; (ndf*4) x 8 x 8

   -- print(d3)
    local d4= d3  - nn.SpatialConvolution(ndf*4, ndf*8, 4, 4, 2, 2, 1, 1) -SpatialBatchNormalization(ndf*8) - nn.LeakyReLU(0.02 ,true)

-- state size ; (ndf*8) x 4 x 4
   local d5= d4 - nn.SpatialConvolution(ndf*8, ndf*8, 4, 4, 2, 2, 1, 1) -SpatialBatchNormalization(ndf*8) -nn.LeakyReLU(0.02,true) --nn.View(torch.LongStorage{bs,ndf*8 *2*2})

--state size ; (ndf*8) x 2x 2
    local d6= d5 -nn.View(torch.LongStorage{bs,ndf*8 *2*2}) -nn.Linear((ndf*8*2*2),1024) -nn.LeakyReLU(0.02,true) -nn.View(bs,1024,1,1)
    

    --print(d4)

    local d7={d6,z2} - nn.JoinTable(2) -nn.SpatialConvolution(2048,2048,1,1) -nn.Dropout(0.2) -nn.LeakyReLU(0.02,true)
    local d8=d7 -nn.SpatialConvolution(2048,1,1,1) -nn.Dropout(0.2) - nn.Sigmoid() - nn.AddConstant(0.0000001,true)-nn.Log() - nn.MulConstant(-1,true) -nn.View(1):setNumInputDims(3)--nn.LeakyReLU(0.02,true) -nn.View(1):setNumInputDims(3)---nn.LogSigmoid() -nn.MulConstant(-1,false)
    netD = nn.gModule({z0,d0},{d8})
    --netD = nn.gModule({d0},{d6})

    return netD

end
--]]
function defineC(nz,nclass)
    local netC=nn.Sequential()

    netC:add(nn.View(-1,nz)):add(nn.Linear(nz,1024)):add(nn.Dropout(0.2)):add(nn.LeakyReLU(0.2,true))
    netC:add(nn.Linear(1024,1024)):add(nn.Dropout(0.2)):add(nn.LeakyReLU(0.2,true))
    netC:add(nn.Linear(1024,nclass))--:add(nn.Sigmoid())
    netC:add(nn.LogSoftMax())


    return netC

end

function defineCL1(input_cnl,nqf,nclass)

    netC=nn.Sequential()
    netC:add(SpatialBatchNormalization(input_cnl)):add(SpatialConvolution(input_cnl, nqf * 2, 4, 4, 2, 2, 1, 1))
netC:add(SpatialBatchNormalization(nqf * 2)):add(nn.LeakyReLU(0.2, true))
-- state size: (ndf*2) x 8 x 8 or 16 x 16
netC:add(SpatialConvolution(nqf*2, nqf*2, 3, 3, 1, 1, 1, 1))
netC:add(SpatialBatchNormalization(nqf * 2)):add(nn.LeakyReLU(0.2, true))
-- state size: (ndf*2) x 8 x 8 or 16 x 16

netC:add(SpatialConvolution(nqf * 2, nqf * 4, 4, 4, 2, 2, 1, 1))
netC:add(SpatialBatchNormalization(nqf * 4)):add(nn.LeakyReLU(0.2, true))
-- state size: (ndf*4) x 4 x 4 or 8 x 8
netC:add(SpatialConvolution(nqf * 4, nqf * 4, 3, 3, 1, 1, 1, 1))
netC:add(SpatialBatchNormalization(nqf * 4)):add(nn.LeakyReLU(0.2, true))
    netC:add(SpatialConvolution(nqf * 4, nclass, 4, 4))
-- state size: nlabel x 1 x 1
-----------------------------------------------------------------
netC:add(nn.SpatialLogSoftMax()):add(nn.View(-1,nclass))
--netC:add(nn.MulConstant(-1,true))
return netC
end

function defineD32_JointProb_U_BiGAN_new(input_nc,ndf,nz)
    -- The first input is Z we upsample that to get to the image size/2 ,the seocnd input is image
    local netD= nil

    local z0=-nn.Identity()
    local z1=z0-SpatialFullConvolution(nz, ndf * 16, 4, 4)- SpatialBatchNormalization(ndf * 16) -nn.LeakyReLU(0.2,true)

-- state size: (16*ndf) x 4 x 4
    local z2= z1- SpatialFullConvolution(ndf * 16, ndf * 8, 4, 4, 2, 2, 1, 1) -SpatialBatchNormalization(ndf * 8) -nn.LeakyReLU(0.2,true)
-- state size: (ndf*8) x 8 x 8
    local z3=z2  - SpatialFullConvolution(ndf * 8, ndf *8, 4, 4, 2, 2, 1, 1) - SpatialBatchNormalization(ndf * 8) -nn.LeakyReLU(0.2,true)
-- state size: (ndf*8) x 16 x 16
   -- local z4= z3 - nn.LeakyReLU(0.2, true) - SpatialFullConvolution(ndf * 2, ndf, 4, 4, 2, 2, 1, 1) - SpatialBatchNormalization(ndf)
-- state size: (ndf) x 32 x 32
  --  local z5= z4 - nn.LeakyReLU(0.2, true) - SpatialFullConvolution(ndf, input_nc, 4, 4, 2, 2, 1, 1) - nn.Tanh()
-- stae size = nc x 64 x 64

    local d0=-nn.Identity()

    --local d0 = {d0_,z5} - nn.JoinTable(2)
-- state size : (nc) x 32 x 32
    local d1_=d0 -nn.SpatialConvolution(input_nc, ndf, 3, 3, 1, 1, 1, 1) - SpatialBatchNormalization(ndf) -nn.LeakyReLU(0.2,true)
    local d1 = d1_ - nn.SpatialConvolution(ndf, ndf*8, 4, 4, 2, 2, 1, 1) - SpatialBatchNormalization(ndf*8) -nn.LeakyReLU(0.2,true)


--  state size ; (ndf*8)*16 x 16
    local d2_0={d1,z3} -nn.CAddTable()
    local d2_1= d2_0 -nn.SpatialConvolution(ndf*8, ndf*8, 3, 3, 1, 1, 1, 1) - SpatialBatchNormalization(ndf*8) -nn.LeakyReLU(0.2,true)
    local d2_2= d2_1 - nn.SpatialConvolution(ndf*8, ndf*8, 4, 4, 2, 2, 1, 1) - SpatialBatchNormalization(ndf*8) -nn.LeakyReLU(0.2,true)


--  state size ; (ndf*8)x 8 x 8
    local d3_0={d2_2,z2} -nn.CAddTable()
    local d3_1=d3_0 -nn.SpatialConvolution(ndf*8, ndf*8, 3, 3, 1, 1, 1, 1) - SpatialBatchNormalization(ndf*8) -nn.LeakyReLU(0.2,true)
    local d3_2= d3_1 - nn.SpatialConvolution(ndf*8, ndf*16, 4, 4, 2, 2, 1, 1) -SpatialBatchNormalization(ndf*16) -nn.LeakyReLU(0.2,true)

-- state size ; (ndf*16) x 4 x 4
    local d4_0={d3_2,z1} -nn.CAddTable()
    local d4_1= d4_0 -nn.SpatialConvolution(ndf*16, ndf*16, 3, 3, 1, 1, 1, 1) - SpatialBatchNormalization(ndf*16) -nn.LeakyReLU(0.2,true)
    local d4_2=d4_1 -nn.SpatialConvolution(ndf*16, ndf*16, 4, 4, 2, 2, 1, 1) -SpatialBatchNormalization(ndf*16) -nn.LeakyReLU(0.2,true)

-- state size ; (ndf*16) x 2 x 2
    local d5_0= d4_2 - nn.SpatialConvolution(ndf*16, ndf*16, 2, 2) -SpatialBatchNormalization(ndf*16) -nn.LeakyReLU(0.2,true)--nn.View(1):setNumInputDims(3)
    local d5_1=d5_0 - nn.SpatialConvolution(ndf*16, 1, 1, 1) -nn.Sigmoid() -nn.AddConstant(0.0000001,true) -nn.Log() -nn.MulConstant(-1,true)
--nn.ReLU(true) -nn.AddConstant(0.0000001,true)-nn.Log()-nn.MulConstant(-1,true)  --nn.LogSigmoid()-nn.MulConstant(-1,true)--nn.LeakyReLU(0.2,true)-nn.View(-1,1):setNumInputDims(3)
-- state size ; (ndf*16) x 1 x 1

   -- local d8= d7 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(2*nz,1,1,1) - nn.LeakyReLU(0.2, true) -nn.View(1):setNumInputDims(3)
    --local d8= d7 - nn.ReLU(true) - nn.SpatialConvolution(2*nz,1,1,1) - nn.LogSigmoid() -nn.MulConstant(-1,false)
   -- local d8= d7 - nn.ReLU(true) - nn.SpatialConvolution(2*nz,1,1,1) - nn.Sigmoid() - nn.AddConstant(0.0000001,false)-nn.Log() - nn.MulConstant(-1,false)


    netD = nn.gModule({z0,d0},{d5_1})

    return netD

end

function weight_visualizer(input_net,display_id)
    disp = require 'display'
    local layer= input_net:get(1)
    local counter=1
    while(layer) do
         local name = torch.type(layer)
         print(name)
          if name:find('Convolution') then
              print(name)
            if layer.weight then
                local lay_W=layer.weight:view(layer.weight:size(1),-1)
                disp.image(lay_W, {win=display_id+counter, title=name..counter})

            end

          end
            counter=counter+1
             layer= input_net:get(counter)
              print(layer)
        end
end