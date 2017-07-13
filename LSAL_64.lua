--
-- Created by IntelliJ IDEA.
-- User: newmoon
-- Date: 4/9/17
-- Time: 2:36 PM
-- To change this template use File | Settings | File Templates.

--

require 'torch'
require 'nn'
require 'optim'
require 'image'
require 'nngraph'
require 'models'
local matio=require'matio'
util = paths.dofile('util/util.lua')



opt = {
   dataset = 'Pascal',       -- svhn / cifar10 / mnist: now we support these three datasets. Users should modify the loading of dataset and get_minibatch function to use their own datasets.
    -- dataset = 'lsun',
   batchSize = 64,--64,
   loadSize = 69,--69,--84,--96,--84,
   fineSize = 64,
   nz =100,-- 200,               -- #  of dim for Z
   ngf = 64,               -- #  of gen filters in first conv layer
   ndf = 64,               -- #  of discrim filters in first conv layer
   nqf=64,
   nThreads = 4,           -- #  of data loading threads to use
   niter = 25,             -- #  of iter at starting learning rate
   lr = 0.0001,-- 0.00005,            -- initial learning rate for adam
   beta1 = 0.5,--0.5,            -- momentum term of adam
   ntrain = math.huge,     -- #  of examples per epoch. math.huge for full dataset
   display = 64,            -- display samples while training. 0 = false
   display_id = 1,--43,        -- display window id.
   gpu = 1,                -- gpu = -1 is CPU mode. gpu=X is GPU mode on GPU X
   name='experiment64',--**********'neat_celebA_saveAll_decay_leak64',--'celebA_saveAll_Wdecay',--'neat_celebA_SaveAll_84crop_mue0-008nu0-099',--'neat_celebA_148_mue0-008nu0-009'
   noise = 'uniform',       -- uniform / normal
   mue=0.008,--*****0.008,             -- best so far mue=0.008 nu=0.009
    nu=0.009,--***0,--0.009,
    gamma = 1,		      -- the coefficient for loss minimization term.  Set to zero for non-conditional LS-GAN as the theorem shows this term can be ignored.
   decay_rate = 0.00005,  -- weight decay: 0.00005
   read_type='float',
   lblchnl=1,

    cudnn=1,}

for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)
if opt.display == 0 then opt.display = false end

opt.manualSeed = torch.random(1, 10000) -- fix seed
print("Random Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)
torch.setnumthreads(1)
torch.setdefaulttensortype('torch.FloatTensor')

-- create data loader
--local DataLoader = paths.dofile('data/data_spaint.lua')
local DataLoader = paths.dofile('data/data.lua')
local data = DataLoader.new(opt.nThreads, opt.dataset, opt)
print("Dataset: " .. opt.dataset, " Size: ", data:size())


local function weights_init(m)
   local name = torch.type(m)
   if name:find('Convolution') then
      m.weight:normal(0.0, 0.02)
      m:noBias()
   elseif name:find('BatchNormalization') then
      if m.weight then m.weight:normal(1.0, 0.02) end
      if m.bias then m.bias:fill(0) end
   end
end

local function weights_init_fcG(m)
   local name = torch.type(m)
   if name:find('Convolution') then
      m.weight:normal(0.0, 0.02)
      m.bias:normal(0.0, 0.01)
   elseif name:find('BatchNormalization') then
      if m.weight then m.weight:normal(1.0, 0.02) end
      if m.bias then m.bias:fill(0) end
   end
end


local function weights_init_fcG1(m)
   local name = torch.type(m)
   if name:find('SpatialConvolution') then
      m.weight:normal(0.0, 0.02)
      m.bias:normal(0.0, 0.01)
   elseif name:find('BatchNormalization') then
      if m.weight then m.weight:normal(1.0, 0.02) end
      if m.bias then m.bias:fill(0) end
   elseif name:find('SpatialFullConvolution')then
       m.weight:normal(0.0, 0.02)
       m:noBias()
  --     elseif name:find('Linear') then
    --    m.weight:normal(0.0, 0.02)
      --  m.bias:normal(0.0, 0.01)

   end
end

local function lbl2onehot(lbl,outClass) -- lbl is in the format of batchsize,w,h
   lbldim=lbl:size()
   if(#lbldim>3)then
      lbl=lbl:view(lbldim[1],lbldim[3],lbldim[4])
   end
   --print('pass 1')
   local batchSize=lbl:size(1)
   --print('batchsize',batchSize)
   local w=lbl:size(2)
   --print('w',w)
   local h=lbl:size(3)
   --print('h',h)
   local one_hot=torch.zeros(outClass,batchSize*w*h)

   local indices=lbl:view(1,-1)
   indices=indices:long()
   --print('***',indices)
   one_hot=one_hot:scatter(1,indices,1)
   one_hot=one_hot:view(outClass,batchSize,w,h)
   one_hot=one_hot:permute(2,1,3,4)

   return one_hot
end
local nc = 3
local nz = opt.nz
local ndf = opt.ndf
local ngf = opt.ngf
local nqf= opt.nqf
local nocl=opt.outClass
local real_label = -1 -- the original one was 1 , we changed that for sake of MarginCriterion
local fake_label = 0

local SpatialBatchNormalization = nn.SpatialBatchNormalization
local SpatialConvolution = nn.SpatialConvolution
local SpatialFullConvolution = nn.SpatialFullConvolution



 netG= defineG_noleak(nc, ngf,nz)
netG:apply(weights_init)
print('NetG is generated')

netD=defineD_JointProb_U_Zsum_lowleak_droplast(nc,ndf,nz,opt.batchSize)
netD:apply(weights_init_fcG1)
print('netD is generated')

netQ=defineQ_noleak(nc,nqf,nz)
netQ:apply(weights_init)
print ('netQ is generated')

print('netG:')
print(netG)
print('netD:')
print(netD)
print('netQ:')
print(netQ)



local criterion = nn.MarginCriterion(0) --set the coresponding y to -1 so it will become loss(x,y)=sum_i(max(0,0-(-1)*x[i]))/x:nElement()
local L2dist=nn.PairwiseDistance(2)
local L1dist=nn.PairwiseDistance(1)
---------------------------------------------------------------------------
optimStateG = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}
optimStateD = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}
optimStateQ = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}



----------------------------------------------------------------------------
local input = torch.Tensor(opt.batchSize, 3, opt.fineSize, opt.fineSize)

local noise = torch.Tensor(opt.batchSize, nz, 1, 1)
local Z_real=torch.Tensor(opt.batchSize, nz, 1, 1)
local input_fakeimg=torch.Tensor(opt.batchSize, 3, opt.fineSize, opt.fineSize)
local label = torch.Tensor(opt.batchSize)
local errD, errG, errQ
local epoch_tm = torch.Timer()
local tm = torch.Timer()
local data_tm = torch.Timer()
local gradd,gradg,gradq
----------------------------------------------------------------------------
if opt.gpu > 0  then
   require 'cunn'
   cutorch.setDevice(opt.gpu)
   input = input:cuda();  noise = noise:cuda();  label = label:cuda();    input_fakeimg=input_fakeimg:cuda(); Z_real=Z_real:cuda();
     if opt.cudnn==1 then
         require 'cudnn'
      netG = util.cudnn(netG); netD = util.cudnn(netD); netQ = util.cudnn(netQ); L2dist=util.cudnn(L2dist); L1dist=util.cudnn(L1dist)
   end

   netD:cuda();           netG:cuda();        netQ:cuda();   criterion:cuda();     L2dist:cuda();     L1dist:cuda();
end

local parametersD, gradParametersD = netD:getParameters()
local parametersG, gradParametersG = netG:getParameters()
local parametersQ,gradparametersQ=netQ:getParameters()

if opt.display then disp = require 'display' end

noise_vis = noise:clone()
tmimg=data:getBatch():clone()
testimg=input:clone():copy(tmimg)

if opt.noise == 'uniform' then
    noise_vis:uniform(-1, 1)
elseif opt.noise == 'normal' then
    noise_vis:normal(0, 1)
end


-- create closure to evaluate f(X) and df/dX of discriminator
local fDx = function(x)
   gradParametersD:zero()

   -- train with real
   data_tm:reset(); data_tm:resume()
   --local real,gtlbl,tmp2 = data:getBatch()
   local real = data:getBatch()

   data_tm:stop()
   input:copy(real)
   label:fill(real_label)

   Z_real=netQ:forward(input):clone()

   if opt.noise == 'uniform' then -- regenerate random noise
       noise:uniform(-1, 1)
   elseif opt.noise == 'normal' then
       noise:normal(0, 1)
   end
   local Z_dist= L1dist:forward({noise:view(opt.batchSize,-1),Z_real:view(opt.batchSize,-1)}):clone():viewAs(label)

   Z_dist:mul(opt.nu)
   local fake = netG:forward (noise)

   input_fakeimg:copy(fake)
   local pdist=L1dist:forward({input:view(opt.batchSize,3* opt.fineSize* opt.fineSize),input_fakeimg:view(opt.batchSize,3* opt.fineSize* opt.fineSize)}):clone():viewAs(label)

   pdist:mul(opt.mue) -- for discriminator this will beome constant doesn't need backward

   local L_r=netD:forward({Z_real,input}):clone()
   L_r=L_r:viewAs(label)
   local L_f=netD:forward({noise,input_fakeimg}):clone():viewAs(label)

   local cost1=pdist+Z_dist+L_r-L_f


   costR = L_r:mean()
   costF = L_f:mean()
   mar = pdist:mean()
   Z_mar=Z_dist:mean()

   local error_hinge = criterion:forward(cost1, label)
   local df_error_hinge = criterion:backward(cost1, label)
   --*********************************
   --********************df_error_hinge:add(0.1)
   --***********************************
   netD:backward({noise,input_fakeimg},-df_error_hinge:viewAs( L_f))

   netD:forward({Z_real,input})
   netD:backward({Z_real,input},df_error_hinge:viewAs( L_r))

    gradd=gradParametersD:clone()


   errD = error_hinge

   return errD, gradParametersD+opt.decay_rate*x
end

local fQz =function(x)
    gradParametersG:zero()
    gradParametersD:zero()
    gradparametersQ:zero()
    --print(#Z_real)
    local outR=netD:forward({Z_real,input})
    errQ=torch.mean(-outR)
    local df_outR=(-1/(opt.batchSize))*outR:clone():fill(1)
    df_Lr=netD:updateGradInput({Z_real,input},df_outR)
    netQ:backward(input,df_Lr[1])
    gradq=gradparametersQ:clone()
    return errQ,gradparametersQ +opt.decay_rate*x

end
-- create closure to evaluate f(X) and df/dX of generator
local fGx = function(x)
   gradParametersG:zero()
   gradParametersD:zero()
   gradparametersQ:zero()
   local outputF=netD:forward({noise,input_fakeimg})

   errG = torch.mean(outputF)
   local df_outF=(1/(opt.batchSize))*outputF:clone():fill(1)
    df_outputF = netD:updateGradInput({noise,input_fakeimg},df_outF)
   netG:backward(noise,df_outputF[2])

   gradg=gradParametersG:clone()
   return errG, gradParametersG--+opt.decay_rate*x
end

-- train

local plot_data_x = {}
local plot_data_z = {}
local plot_win_x
local plot_win_z

local plot_config_marX = {
  title = opt.name,
  labels = {"epoch", "delta-X"},
  ylabel = "delta X",
}
local plot_config_marZ=
{
    title = opt.name,
  labels = {"epoch", "delta-Z"},
  ylabel = "delta Z",
}
paths.mkdir(opt.name)

local gradqlist=torch.Tensor(1,1)
local graddlist=torch.Tensor(1,1)
local gradglist=torch.Tensor(1,1)
local margZlist=torch.Tensor(1,1)
local margXlist=torch.Tensor(1,1)
local ErrDlist=torch.Tensor(1,1)
local ErrGlist=torch.Tensor(1,1)
local ErrQlist=torch.Tensor(1,1)
 flag=0

for epoch = 1, opt.niter do
  \

-- display plot vars

   epoch_tm:reset()
   counter = -1
   for i = 1, math.min(data:size(), opt.ntrain), opt.batchSize do
      tm:reset()

      optim.adam(fDx, parametersD, optimStateD)-- original

      optim.adam(fGx, parametersG, optimStateG)

      optim.adam(fQz,parametersQ,optimStateQ)

      counter = counter+1
      if counter % 10 == 0 and opt.display then -


          local fake = netG:forward(noise_vis):clone()

           local z_=netQ:forward(testimg)


          local Reconstruct=netG:forward(z_):clone()




          local real = data:getBatch()
          disp.image(fake, {win=opt.display_id, title=opt.name})
          --disp.image(testimg, {win=opt.display_id +1, title=opt.name})
          disp.image(Reconstruct,{win=opt.display_id +2, title=opt.name..'_recunstruct'})



            local serial_batches=opt.serial_batches

            if counter%20==0 then
            local image_out_fake = nil
            local image_out_recons=nil
            local image_real_train=nil



                print('save to the disk')

                local sqrsize=torch.sqrt(fake:size(1))
                for i1=1,sqrsize do
                    local img_tmp_fake=nil
                    local img_tmp_recon=nil
                    local img_tmp_real=nil
                    for i2=1,sqrsize do
                        if img_tmp_fake==nil then img_tmp_fake=fake[(i1-1)*sqrsize+i2]
                            img_tmp_recon=Reconstruct[(i1-1)*sqrsize+i2]
                            img_tmp_real=tmimg[(i1-1)*sqrsize+i2]
                            --print('done1')
                        else
                            img_tmp_fake=torch.cat(img_tmp_fake,fake[(i1-1)*sqrsize+i2],3)
                            img_tmp_recon=torch.cat(img_tmp_recon,Reconstruct[(i1-1)*sqrsize+i2],3)
                            img_tmp_real=torch.cat(img_tmp_real,tmimg[(i1-1)*sqrsize+i2],3)
                        end
                    end
                    if image_out_fake==nil then
                        image_out_fake=img_tmp_fake
                        image_out_recons=img_tmp_recon
                        image_real_train=img_tmp_real


                    else
                    image_out_fake=torch.cat(image_out_fake,img_tmp_fake,2)
                    image_out_recons=torch.cat(image_out_recons,img_tmp_recon,2)
                    image_real_train= torch.cat(image_real_train,img_tmp_real,2)

                    end
                end

            image.save(paths.concat(opt.name ,'epoch'..epoch..'i' .. counter .. '_fake.png'), image_out_fake:add(1):div(2))
            image.save(paths.concat(opt.name ,'epoch'..epoch..'i' .. counter .. '_reconstruct.png'), image_out_recons:add(1):div(2))
            image.save(paths.concat(opt.name , '_real.png'), image_real_train:add(1):div(2))



            end
          print(opt.name)
      end



      if ((i-1) / opt.batchSize) % 1 == 0 then --gradE: %5f,torch.mean(torch.abs(gradparametersE)),

      if flag==0 then

          graddlist:fill(torch.mean(torch.abs(gradd)))
          gradglist[1][1]=torch.mean(torch.abs(gradg))
          gradqlist[1][1]=torch.mean(torch.abs(gradq))
          margZlist[1][1]=Z_mar
          margXlist[1][1]=margXlist
          ErrDlist[1][1]=errD
          ErrGlist[1][1]=errG
          ErrQlist[1][1]=errQ
          flag=1

      else



          graddlist=torch.cat(graddlist,torch.FloatTensor(1,1):fill(torch.mean(torch.abs(gradd))),2)
          gradglist=torch.cat(gradglist,torch.FloatTensor(1,1):fill(torch.mean(torch.abs(gradg))),2)
          gradqlist=torch.cat(gradqlist,torch.FloatTensor(1,1):fill(torch.mean(torch.abs(gradq))),2)-- torch.mean(torch.abs(gradq))
          margZlist=torch.cat(margZlist,torch.FloatTensor(1,1):fill(Z_mar),2)

          margXlist=torch.cat(margXlist,torch.FloatTensor(1,1):fill(mar),2)--margXlist
          ErrDlist=torch.cat(ErrDlist,torch.FloatTensor(1,1):fill(errD),2)
          ErrGlist=torch.cat(ErrGlist,torch.FloatTensor(1,1):fill(errG),2)
          ErrQlist=torch.cat(ErrQlist,torch.FloatTensor(1,1):fill(errQ),2)--[1][1]=errQ

      end




         print(('Epoch: [%d][%8d / %8d]\t Time: %.3f  DataTime: %.3f  '
                   .. '  Err_G: %.9f  Err_D: %.9f Err_Q:%9f  costR:%.7f   costF:%.7f   meanD:%.5f mean_Dz:%4f gradD:%.9f  gradG:%.9f gradQ:%.9f'):format( -- gradD:%.5f   gradG:%.7f, gradQ:%.7f
                 epoch, ((i-1) / opt.batchSize),
                 math.floor(math.min(data:size(), opt.ntrain) / opt.batchSize),
                 tm:time().real, data_tm:time().real,
                 errG and errG or -1, errD and errD or -1,errQ and errQ or -1, costR, costF, mar,Z_mar,torch.mean(torch.abs(gradd)),torch.mean(torch.abs(gradg)),torch.mean(torch.abs(gradq))))--gradParametersG gradParametersG
                --torch.mean(torch.abs(gradParametersD)), torch.mean(torch.abs(gradParametersG)),torch.mean(torch.abs(gradParametersQ))

      local curItInBatch = ((i-1) / opt.batchSize)
      local totalItInBatch = math.floor(math.min(data:size(), opt.ntrain) / opt.batchSize)

      local plot_vals_x = { epoch + curItInBatch / totalItInBatch }
      local plot_vals_z = { epoch + curItInBatch / totalItInBatch }
            plot_vals_z[#plot_vals_z+1]=Z_mar

           -- for k, v in ipairs(opt.display_plot) do
           --   if loss[v] ~= nil then
            plot_vals_x[#plot_vals_x + 1] = mar
           --  end
          --  end
            -- update display plot
            if opt.display then
              table.insert(plot_data_x, plot_vals_x)
              plot_config_marX.win = plot_win_x
              plot_win_x = disp.plot(plot_data_x, plot_config_marX)

              table.insert(plot_data_z,plot_vals_z)
              plot_config_marZ.win=plot_win_z
              plot_win_z =disp.plot(plot_data_z,plot_config_marZ)
            end

      end
      if(counter%100==0)then
          torch.save(opt.name ..'/'.. opt.name .. '_' .. 'margin_Z.t7', plot_data_z)
          torch.save(opt.name ..'/'.. opt.name .. '_' .. epoch .. '_margin_X.t7', plot_data_x)
          --print(plot_data_z)
          matio.save(opt.name ..'/'.. opt.name .. '_' .. epoch .. '_margin_Z.mat',margZlist)
          matio.save(opt.name ..'/'.. opt.name .. '_' .. epoch .. '_margin_X.mat',margXlist)
          matio.save(opt.name ..'/'.. opt.name .. '_' .. epoch .. 'graddlist.mat',graddlist)
          matio.save(opt.name ..'/'.. opt.name .. '_' .. epoch .. 'gradglist.mat',gradglist)
          matio.save(opt.name ..'/'.. opt.name .. '_' .. epoch .. 'gradqlist.mat',gradqlist)
          matio.save(opt.name ..'/'.. opt.name .. '_' .. epoch .. 'ErrDlist.mat',ErrDlist)
          matio.save(opt.name ..'/'.. opt.name .. '_' .. epoch .. 'ErrGlist.mat',ErrGlist)
          matio.save(opt.name ..'/'.. opt.name .. '_' .. epoch .. 'ErrQlist.mat',ErrQlist)
      end

   end
   --paths.mkdir('Cityscape64_checkpoint')--pascal64_U
   --('Celeb_LSAL_checkpoint_lowleak2')--
   parametersD, gradParametersD = nil, nil -- nil them to avoid spiking memory
   parametersG, gradParametersG = nil, nil
   parametersQ, gradparametersQ = nil, nil
   gradd,gradg,gradq=nil,nil,nil
   torch.save(opt.name ..'/'.. opt.name .. '_' .. epoch .. '_net_G.t7', netG:clearState())
   torch.save(opt.name ..'/'.. opt.name .. '_' .. epoch .. '_net_D.t7', netD:clearState())
   torch.save(opt.name .. '/'..opt.name .. '_' .. epoch .. '_net_Q.t7', netQ:clearState())
   parametersD, gradParametersD = netD:getParameters() -- reflatten the params and get them
   parametersG, gradParametersG = netG:getParameters()
   parametersQ, gradparametersQ=netQ:getParameters()
   print(('End of epoch %d / %d \t Time Taken: %.3f'):format(
            epoch, opt.niter, epoch_tm:time().real))
end

-- DATA_ROOT=Pascal_33c_84/Smap_mat dataset=Pascal th /home/newmoon/GAN/dcgan.torch-master/Pascal_64_CondG_CondD_SM.lua

-- DATA_ROOT=Pascal_21_84/Smap_mat dataset=Pascal th /home/newmoon/GAN/dcgan.torch-master/Pascal64_U_Wbias_BillnUP.lua
--DATA_ROOT=Pascal_33c_84/Smap_mat dataset=Pascal th /home/newmoon/GAN/dcgan.torch-master/Pascal64_U_Wbias_BillnUP.lua
--DATA_ROOT=Pascal_33c_84/Smap_mat dataset=Pascal th /home/newmoon/GAN/dcgan.torch-master/Pascal64_U_Wbias_HalfBiHalfup_showSM.lua
--DATA_ROOT=CityScape84_34cls/Smap_mat dataset=Pascal th /home/newmoon/GAN/dcgan.torch-master/Pascal64_U_Wbias_HalfBiHalfup_showSM.lua
--DATA_ROOT=/home/newmoon/GAN/dcgan.torch-master/celebA dataset=folder th /home/newmoon/GAN/LSAL/LSAL_dcheck.lua
--DATA_ROOT=/home/newmoon/GAN/dcgan.torch-master/celebA dataset=folder th /home/newmoon/GAN/LSAL/LSAL_lowleak.lua

--DATA_ROOT=/home/newmoon/GAN/dcgan.torch-master/celebA dataset=folder th /home/newmoon/GAN/LSAL/mue0-008nu0-009recons.lua
--DATA_ROOT=/home/liheng/celeba_neat/84 dataset=folder th /home/newmoon/GAN/LSAL/neat_celebA_saveimg_LSAL.lua
--DATA_ROOT=/home/liheng/celeba_neat/148 dataset=folder th /home/newmoon/GAN/LSAL/neat_celebA_saveimg_LSAL.lua
--DATA_ROOT=/home/liheng/celeba_neat/84 dataset=folder th /home/newmoon/GAN/LSAL/neat_celebA_save_everything_LSAL.lua
----DATA_ROOt=/home/gqi/GAN/LSGAN/lsun/train dataset=lsun th /home/newmoon/GAN/LSAL/LSUN_mue0_08nu0_1.lua
--DATA_ROOT=/home/newmoon/GAN/dcgan.torch-master/celebA dataset=folder th /home/newmoon/GAN/LSAL/celebA_saveAll_LSAL_Wdecay.lua
--DATA_ROOT=/home/liheng/celeba_neat/84 dataset=folder th /home/newmoon/GAN/LSAL/celebA_saveAll_LSAL_Wdecay.lua
