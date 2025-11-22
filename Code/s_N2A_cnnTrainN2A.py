
# s_DtiNet_cnnTrain.py
#
# QT 2019

# %% 在所有导入之前设置环境变量来解决内存碎片化
import os
import sys
import time
# 设置CUDA内存分配器为异步模式，解决内存碎片化
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['TF_ENABLE_GPU_GARBAGE_COLLECTION'] = '1'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
# 禁用TensorFlow的JIT编译，减少内存使用
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0'

# %% load modules with delay
import time
import argparse
import gc
import scipy.io as sio
import numpy as np
import nibabel as nb
import glob
import tensorflow
print(f"🚀 TensorFlow version: {tensorflow.__version__}")
print("🔧 GPU memory optimization enabled: TF_GPU_ALLOCATOR=cuda_malloc_async")

import munet_res
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
import qtlib as qtlib
import tensorflow as tf
from tensorflow.keras import backend as K

# %% 添加命令行参数解析
parser = argparse.ArgumentParser(description='Train Noise2Average model')
parser.add_argument('--subject_idx', type=int, default=0, 
                    help='Subject index to train (0-9, default: 0)')
parser.add_argument('--gpu_id', type=str, default='0',
                    help='GPU ID to use (default: 0)')
parser.add_argument('--gpu_memory_limit', type=int, default=16000,
                    help='GPU memory limit in MB (default: None for growth mode)')
args = parser.parse_args()

# 设置GPU
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

# %% 配置TensorFlow GPU显存使用 (适配TensorFlow 2.15.0)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        gpu = gpus[0]
        
        if args.gpu_memory_limit:
            # 方法1: 设置显存限制
            tf.config.experimental.set_memory_growth(gpu, False)
            tf.config.experimental.set_virtual_device_configuration(
                gpu,
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=args.gpu_memory_limit)]
            )
            print(f"✅ GPU memory limit set to {args.gpu_memory_limit} MB ({args.gpu_memory_limit/1024:.1f} GB)")
        else:
            # 方法2: 启用内存增长 (按需分配，推荐)
            tf.config.experimental.set_memory_growth(gpu, True)
            print("✅ GPU memory growth enabled - will allocate as needed")
            
    except RuntimeError as e:
        print(f"❌ GPU configuration error: {e}")
        print("Note: GPU configuration must be set before any operations")
    except Exception as e:
        print(f"⚠️ GPU configuration warning: {e}")
        print("Continuing with default GPU settings...")

def print_gpu_status():
    """详细的GPU状态打印"""
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.used,memory.total,memory.free,utilization.gpu,temperature.gpu', 
                               '--format=csv,nounits,noheader'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for i, line in enumerate(lines):
                values = [v.strip() for v in line.split(',')]
                if len(values) >= 6:
                    name, mem_used, mem_total, mem_free, gpu_util, temp = values[:6]
                    mem_used_int = int(mem_used)
                    mem_total_int = int(mem_total)
                    mem_free_int = int(mem_free)
                    usage_percent = (mem_used_int/mem_total_int)*100
                    
                    print(f"🖥️  GPU {i}: {name}")
                    print(f"   💾 Memory: {mem_used}/{mem_total}MB ({usage_percent:.1f}%) | Free: {mem_free}MB")
                    print(f"   ⚡ Util: {gpu_util}% | 🌡️  Temp: {temp}°C")
                    
                    # 内存碎片化和警告检测
                    if mem_free_int < 2000 and usage_percent < 85:
                        print(f"   ⚠️  Possible memory fragmentation detected!")
                    elif mem_free_int < 1000:
                        print(f"   🚨 Low memory warning: {mem_free_int}MB free")
    except Exception as e:
        print(f"Could not get GPU status: {e}")

def aggressive_memory_cleanup():
    """积极的内存清理策略"""
    print("🧹 Performing aggressive memory cleanup...")
    
    # 1. 清理Keras/TensorFlow缓存
    K.clear_session()
    tf.keras.backend.clear_session()
    
    # 2. Python垃圾回收
    collected = gc.collect()
    print(f"   🗑️  Garbage collected: {collected} objects")
    
    # 3. GPU内存统计重置
    if gpus:
        try:
            tf.config.experimental.reset_memory_stats(gpus[0])
            print("   📊 GPU memory stats reset")
        except:
            pass
    
    # 4. 创建临时tensor触发内存重新分配（碎片整理）
    try:
        temp_tensor = tf.random.normal([200, 200])
        del temp_tensor
        gc.collect()
        print("   🔧 Memory defragmentation performed")
    except:
        pass
    
    print("   ✅ Aggressive cleanup completed")

def smart_memory_management(current_epoch, total_epochs):
    """智能内存管理策略"""
    try:
        # 获取当前内存使用情况
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.free,memory.total', 
                               '--format=csv,nounits,noheader'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            mem_used, mem_free, mem_total = result.stdout.strip().split(', ')
            mem_used_int = int(mem_used)
            mem_free_int = int(mem_free)
            mem_total_int = int(mem_total)
            usage_percent = (mem_used_int / mem_total_int) * 100
            
            print(f"💾 Memory Status: {mem_used_int}MB used / {mem_total_int}MB total ({usage_percent:.1f}%)")
            print(f"   🆓 Available: {mem_free_int}MB")
            
            # 策略1: 如果可用内存少于2GB，执行清理
            if mem_free_int < 2048:
                print(f"⚠️  Low free memory ({mem_free_int}MB), performing cleanup...")
                aggressive_memory_cleanup()
                return True
            
            # 策略2: 如果使用率超过85%，执行清理
            elif usage_percent > 85:
                print(f"⚠️  High memory usage ({usage_percent:.1f}%), performing cleanup...")
                aggressive_memory_cleanup()
                return True
            
            # 策略3: 每5个epoch进行预防性清理
            elif current_epoch > 0 and current_epoch % 5 == 0:
                print(f"🔧 Preventive memory cleanup at epoch {current_epoch}...")
                aggressive_memory_cleanup()
                return True
                
        return False
    except Exception as e:
        print(f"Memory management error: {e}")
        return False

def emergency_memory_recovery():
    """紧急内存恢复策略"""
    print("🚨 Emergency memory recovery initiated...")
    
    # 1. 最激进的清理
    K.clear_session()
    tf.keras.backend.clear_session()
    
    # 2. 强制垃圾回收多次
    for i in range(3):
        collected = gc.collect()
        print(f"   🗑️  GC round {i+1}: {collected} objects")
    
    # 3. 重置GPU内存
    if gpus:
        try:
            tf.config.experimental.reset_memory_stats(gpus[0])
            # 尝试创建小tensor来触发内存重组
            for size in [50, 100, 150]:
                try:
                    temp = tf.random.normal([size, size])
                    del temp
                except:
                    break
            gc.collect()
            print("   🔧 Emergency memory defragmentation completed")
        except Exception as e:
            print(f"   ⚠️  GPU reset failed: {e}")
    
    print("   ✅ Emergency recovery completed")

# %% 优化的显存清理函数
def clear_memory():
    """标准内存清理"""
    K.clear_session()
    tf.keras.backend.clear_session()
    gc.collect()
    
    # 如果使用GPU，清理GPU缓存
    if gpus:
        try:
            tf.config.experimental.reset_memory_stats(gpus[0])
        except:
            pass
    
    print("   📝 Standard memory cleared")

# %% 日志记录函数
def write_log(log_file, message):
    """写入日志信息"""
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    log_message = f"[{timestamp}] {message}\n"
    
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(log_message)
    
    print(log_message.strip())

# %% set up path
dpRoot = '../Data'
os.chdir(dpRoot)

# %% subjects
subjects = [f'sub_001']

# 检查subject索引是否有效
if args.subject_idx >= len(subjects) or args.subject_idx < 0:
    raise ValueError(f"Subject index {args.subject_idx} is out of range. Valid range: 0-{len(subjects)-1}")

print(f"🎯 Training subject {args.subject_idx}: {subjects[args.subject_idx]}")

nbatch = 1
nepoch = 20
niter = 2
shouldtrain = True
expname = f'N2A-mem{args.gpu_memory_limit or "auto"}MB'

# 显示初始GPU状态
print("\n" + "="*60)
print("🚀 INITIAL GPU STATUS")
print("="*60)
print_gpu_status()

# %% load data with memory optimization
if shouldtrain:
    for ch in range(0,1):
        # 只训练指定的subject
        ii = args.subject_idx
        sj = os.path.basename(subjects[ii])
        print(f"\n🔄 Processing subject: {sj}")
        
        # 创建日志文件路径
        dpSub = os.path.join(dpRoot, sj)
        dpCnn = os.path.join(dpSub, expname) 
        if not os.path.exists(dpCnn):
            os.makedirs(dpCnn)
            print('📁 Created directory')
        
        log_file = os.path.join(dpCnn, f'training_log_ch{ch}.txt')
        
        # 初始化日志
        write_log(log_file, f"=== Training Started for {sj} Channel {ch} ===")
        write_log(log_file, f"GPU ID: {args.gpu_id}")
        write_log(log_file, f"GPU Memory Limit: {args.gpu_memory_limit or 'Auto-growth'}")
        write_log(log_file, f"TensorFlow Version: {tensorflow.__version__}")
        write_log(log_file, f"Memory Optimization: cuda_malloc_async enabled")
        write_log(log_file, f"Block Size: 80x80x80 (Large blocks - high memory usage expected)")
        write_log(log_file, f"Batch Size: {nbatch}, Epochs per Iteration: {nepoch}, Total Iterations: {niter}")
        
        # 数据文件路径
        fpImg1 = os.path.join(dpSub,'crop_T1_GT_pad_RamTransVerySmallSplineNoise1_FLIRTAffine2GT_withmask.nii.nii.gz')
        fpImg2 = os.path.join(dpSub,'crop_T1_GT_pad_RamTransVerySmallSplineNoise2_FLIRTAffine2GT_withmask.nii.nii.gz') 
        fpMask = os.path.join(dpSub,'T1_mask.nii.gz')

        print("📂 Loading images...")
        write_log(log_file, "Loading data files...")
        
        # 监控内存使用
        print("💾 Before loading images:")
        print_gpu_status()
        
        img1 = nb.load(fpImg1).get_fdata()
        img2 = nb.load(fpImg2).get_fdata()
        mask = nb.load(fpMask).get_fdata()
        
        print("💾 After loading images:")
        print_gpu_status()
        
        if len(mask.shape)==3:
            mask = np.expand_dims(mask, 3)    
        if len(img1.shape)==3:
            img1 = np.expand_dims(img1, 3)    
        if len(img2.shape)==3:
            img2 = np.expand_dims(img2, 3)    
        
        write_log(log_file, f"Image shapes: img1={img1.shape}, img2={img2.shape}, mask={mask.shape}")
        print(f"📐 Image shapes: img1={img1.shape}, img2={img2.shape}, mask={mask.shape}")

        # 使用80x80x80的块大小 - 内存使用量大
        ind_block, ind_brain = qtlib.block_ind(mask, sz_block=80, sz_pad=0)
        write_log(log_file, f"Block extraction: {len(ind_block)} blocks of size 80x80x80")
        print(f"🧮 Block extraction: {len(ind_block)} blocks of size 80x80x80")
        
        # get stats
        img1_mean = np.mean(img1[mask > 0.5])
        img1_std = np.std(img1[mask > 0.5])
        img2_mean = np.mean(img2[mask > 0.5])
        img2_std = np.std(img2[mask > 0.5])
        
        write_log(log_file, f"Image statistics - img1: mean={img1_mean:.4f}, std={img1_std:.4f}")
        write_log(log_file, f"Image statistics - img2: mean={img2_mean:.4f}, std={img2_std:.4f}")
        
        img1_norm = (img1 - img1_mean) / img1_std 
        img2_norm = (img2 - img2_mean) / img2_std 
        
        print("🧮 Extracting blocks...")
        img1_norm_block = qtlib.extract_block(img1_norm * mask, ind_block)
        img2_norm_block = qtlib.extract_block(img2_norm * mask, ind_block)
        mask_block = qtlib.extract_block(mask, ind_block)
        
        print("💾 After block extraction:")
        print_gpu_status()
        
        imgavg = (img1 + img2) / 2.0

        for jj in np.arange(0, niter):
            iter_start_time = time.time()
            print(f"\n🔄 Starting Iteration {jj}")
            write_log(log_file, f"=== Starting Iteration {jj} ===")
            
            # 在每次迭代开始时清理显存
            print(f"🧹 Before cleanup (Iteration {jj}):")
            print_gpu_status()
            aggressive_memory_cleanup()
            print(f"✅ After cleanup (Iteration {jj}):")
            print_gpu_status()
            
            imgavg_block = qtlib.extract_block(imgavg, ind_block)

            imgavg1_norm_block = (imgavg_block - img1_mean) / img1_std
            imgavg2_norm_block = (imgavg_block - img2_mean) / img2_std
            
            imgres1_block = (imgavg1_norm_block - img1_norm_block) 
            imgres2_block = (imgavg2_norm_block - img2_norm_block) 
            
            imgres1_block = np.concatenate((imgres1_block, mask_block), axis=-1)
            imgres2_block = np.concatenate((imgres2_block, mask_block), axis=-1)

            # 明确清理临时变量
            del imgavg_block, imgavg1_norm_block, imgavg2_norm_block
            gc.collect()
            
            img_block_all = 0 # clear
            imgres_block_all = 0
            mask_block_all = 0
        
            # get block for image1        
            img_block_all = np.concatenate((img1_norm_block, img2_norm_block), axis=0)
            imgres_block_all = np.concatenate((imgres1_block, imgres2_block), axis=0)
            mask_block_all = np.concatenate((mask_block, mask_block), axis=0)
            
            # 清理临时变量
            del imgres1_block, imgres2_block
            gc.collect()
            
            # %% flip left right to augment data
            print("🔄 Applying data augmentation...")
            tmp = np.flip(img_block_all, 1)
            img_block_all = np.concatenate((img_block_all, tmp), axis=0)
            del tmp
            gc.collect()
            
            tmp = np.flip(imgres_block_all, 1)
            imgres_block_all = np.concatenate((imgres_block_all, tmp), axis=0)
            del tmp
            gc.collect()
            
            tmp = np.flip(mask_block_all, 1)
            mask_block_all = np.concatenate((mask_block_all, tmp), axis=0)
            del tmp
            gc.collect()

            write_log(log_file, f"Data preparation completed. Training blocks: {img_block_all.shape[0]}")
            print(f"📊 Training data prepared: {img_block_all.shape[0]} blocks")

            print("💾 After data augmentation:")
            print_gpu_status()

            # %% set up model with memory monitoring
            dtnet = 0 # clear
            fpCp = os.path.join('../PretrainModel', 'superv_lr1e4_ep20.h5') 
            model_load_start = time.time()
            write_log(log_file, 'Loading pretrained model...')
            
            print("🏗️ Before model loading:")
            print_gpu_status()
            
            try:
                dtnet = load_model(fpCp, custom_objects={'mean_squared_error_weighted': qtlib.mean_squared_error_weighted})
                model_load_time = time.time() - model_load_start
                write_log(log_file, f'Model loaded successfully in {model_load_time:.2f} seconds')
                print(f"✅ Model loaded in {model_load_time:.2f}s")
                
                print("💾 After model loading:")
                print_gpu_status()
                
            except Exception as model_error:
                write_log(log_file, f'❌ Model loading failed: {model_error}')
                print(f"❌ Model loading failed: {model_error}")
                raise model_error
            
            # %% set up adam optimizer
            adam_opt = Adam(lr=0.00005, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
            dtnet.compile(loss = qtlib.mean_squared_error_weighted, optimizer = adam_opt)
                
            # train dtnet with enhanced memory management
            for mm in np.arange(0, nepoch):
                starttime = time.time()
                print(f"📚 Starting Epoch {mm}/{nepoch-1} (Iteration {jj})")
                write_log(log_file, f'--- Starting Epoch {mm} of Iteration {jj} ---')
                
                # 智能内存管理
                memory_cleaned = smart_memory_management(mm, nepoch)
                if memory_cleaned:
                    print("   🔧 Memory management performed")
                
                # 每5个epoch显示详细GPU状态
                if mm % 5 == 0 or mm == 0:
                    print(f"📊 Detailed GPU status at epoch {mm}:")
                    print_gpu_status()
                
                fnCp = 'n2a_iter' + str(jj) + '_lr1e5_ep' + str(mm) + f'_ch{ch}'
                fpCp = os.path.join(dpCnn, fnCp + '.h5')     
                
                # 使用更节省内存的checkpoint策略
                checkpoint = ModelCheckpoint(
                    fpCp, 
                    monitor='val_loss', 
                    save_best_only=True,
                    save_weights_only=False,  # 保存完整模型以便后续加载
                    verbose=0
                )
                
                write_log(log_file, f'Training data shape: {img_block_all.shape}')
                
                try:
                    # 训练前小清理
                    if mm > 0:
                        gc.collect()
                    
                    history = dtnet.fit(x = [img_block_all, mask_block_all], 
                                        y = imgres_block_all, 
                                        batch_size = nbatch, 
                                        validation_split=0.2,
                                        epochs = 1, 
                                        callbacks = [checkpoint],
                                        verbose = 1, 
                                        shuffle = True) 
                    
                    epochtime = time.time() - starttime
                    
                    # 记录训练时间和损失
                    train_loss = history.history['loss'][0] if history.history['loss'] else 'N/A'
                    val_loss = history.history['val_loss'][0] if history.history['val_loss'] else 'N/A'
                    
                    log_message = f"Iter {jj}, Epoch {mm} completed in {epochtime:.2f}s - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
                    write_log(log_file, log_message)
                    print(f"✅ {log_message}")
                    
                    # save loss
                    fpLoss = os.path.join(dpCnn, fnCp + '.mat') 
                    sio.savemat(fpLoss, {
                        'loss_train':history.history['loss'], 
                        'loss_val':history.history['val_loss'],
                        'memory_optimization': 'cuda_malloc_async',
                        'block_size': 80,
                        'gpu_memory_limit': args.gpu_memory_limit
                    })
                    
                    # 训练后立即清理
                    del history
                    gc.collect()
                    
                except tf.errors.ResourceExhaustedError as e:
                    write_log(log_file, f'❌ OOM Error at epoch {mm}: {str(e)}')
                    print(f"❌ GPU OOM at epoch {mm} - attempting recovery...")
                    
                    # 紧急内存恢复
                    emergency_memory_recovery()
                    print("💾 After emergency recovery:")
                    print_gpu_status()
                    
                    # 尝试重新训练该epoch
                    try:
                        print("🔄 Retrying training with emergency cleanup...")
                        history = dtnet.fit(x = [img_block_all, mask_block_all], 
                                            y = imgres_block_all, 
                                            batch_size = nbatch, 
                                            validation_split=0.2,
                                            epochs = 1, 
                                            callbacks = [checkpoint],
                                            verbose = 1, 
                                            shuffle = True) 
                        print("✅ Emergency retry successful!")
                        write_log(log_file, f"Emergency retry successful for epoch {mm}")
                        
                        epochtime = time.time() - starttime
                        train_loss = history.history['loss'][0] if history.history['loss'] else 'N/A'
                        val_loss = history.history['val_loss'][0] if history.history['val_loss'] else 'N/A'
                        log_message = f"Iter {jj}, Epoch {mm} (RETRY) completed in {epochtime:.2f}s - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
                        write_log(log_file, log_message)
                        
                        # save loss
                        fpLoss = os.path.join(dpCnn, fnCp + '.mat') 
                        sio.savemat(fpLoss, {'loss_train':history.history['loss'], 'loss_val':history.history['val_loss']})
                        
                        del history
                        gc.collect()
                        
                    except Exception as retry_error:
                        write_log(log_file, f'❌ Emergency retry also failed: {retry_error}')
                        print(f"❌ Emergency retry failed, skipping epoch {mm}")
                        print("💡 Consider:")
                        print("   1. Reducing block size from 80 to 64 or 48")
                        print("   2. Increasing GPU memory limit")
                        print("   3. Using gradient accumulation")
                        continue
                
                except Exception as e:
                    write_log(log_file, f'❌ Training error at epoch {mm}: {e}')
                    print(f"❌ Training error at epoch {mm}: {e}")
                    raise e
                
                # 删除旧checkpoint文件
                if mm >= 1:
                    fnCp_old = 'n2a_iter' + str(jj) + '_lr1e5_ep' + str(mm - 1) + f'_ch{ch}'
                    fpCp_old = os.path.join(dpCnn, fnCp_old + '.h5') 
                    if os.path.exists(fpCp_old):
                        os.remove(fpCp_old)
                        write_log(log_file, f"🗑️  Removed old checkpoint: {fnCp_old}.h5")
            
                # %% apply fine tuned munet for evaluation
                if (mm==(nepoch-1)) or mm % 5 == 0:
                    eval_start_time = time.time()
                    print(f"📊 Starting evaluation for Epoch {mm}...")
                    write_log(log_file, f"Starting evaluation for Epoch {mm}...")
                    
                    print(f"💾 Before evaluation (Epoch {mm}):")
                    print_gpu_status()
                    
                    dpPred = os.path.join(dpSub, expname,'eval') 
                    if not os.path.exists(dpPred):
                        os.makedirs(dpPred)
                        write_log(log_file, '📁 Created evaluation directory')
                        
                    try:
                        # Evaluation with memory management - predict in small batches
                        print("   🔍 Evaluating image1...")
                        img_tmp = img1_norm_block
                        mask_tmp = mask_block
                        pred_tmp = np.zeros(img_tmp.shape)
                        
                        num_blocks = img_tmp.shape[0]
                        for kk in range(num_blocks):
                            if kk % 5 == 0:  # Progress indicator every 5 blocks
                                print(f"      📊 Block {kk+1}/{num_blocks} ({(kk+1)/num_blocks*100:.1f}%)")
                            tmp = dtnet.predict([img_tmp[kk:kk+1, :, :, :, :], mask_tmp[kk:kk+1, :, :, :, :]], verbose=0) 
                            pred_tmp[kk:kk+1, :, :, :, :] = tmp[:, :, :, :, 0:1] + img_tmp[kk:kk+1, :, :, :, :]
                        img1_block_pred = pred_tmp
                        
                        print("   🔍 Evaluating image2...")
                        img_tmp = img2_norm_block
                        mask_tmp = mask_block
                        pred_tmp = np.zeros(img_tmp.shape)
                        for kk in range(num_blocks):
                            tmp = dtnet.predict([img_tmp[kk:kk+1, :, :, :, :], mask_tmp[kk:kk+1, :, :, :, :]], verbose=0) 
                            pred_tmp[kk:kk+1, :, :, :, :] = tmp[:, :, :, :, 0:1] + img_tmp[kk:kk+1, :, :, :, :]
                        img2_block_pred = pred_tmp            
                        
                        print("   🔄 Evaluating flipped image1...")
                        img_tmp = np.flip(img1_norm_block, 1)
                        mask_tmp = np.flip(mask_block, 1)
                        pred_tmp = np.zeros(img_tmp.shape)
                        for kk in range(num_blocks):
                            tmp = dtnet.predict([img_tmp[kk:kk+1, :, :, :, :], mask_tmp[kk:kk+1, :, :, :, :]], verbose=0) 
                            pred_tmp[kk:kk+1, :, :, :, :] = tmp[:, :, :, :, 0:1] + img_tmp[kk:kk+1, :, :, :, :]
                        img1_block_pred_flip = np.flip(pred_tmp, 1)              

                        print("   🔄 Evaluating flipped image2...")
                        img_tmp = np.flip(img2_norm_block, 1)
                        mask_tmp = np.flip(mask_block, 1)
                        pred_tmp = np.zeros(img_tmp.shape)
                        for kk in range(num_blocks):
                            tmp = dtnet.predict([img_tmp[kk:kk+1, :, :, :, :], mask_tmp[kk:kk+1, :, :, :, :]], verbose=0) 
                            pred_tmp[kk:kk+1, :, :, :, :] = tmp[:, :, :, :, 0:1] + img_tmp[kk:kk+1, :, :, :, :]
                        img2_block_pred_flip = np.flip(pred_tmp, 1)   

                        print("   🧮 Combining results...")
                        img1_block_pred = (img1_block_pred + img1_block_pred_flip) / 2.0
                        img2_block_pred = (img2_block_pred + img2_block_pred_flip) / 2.0
                        
                        img1_block_pred = (img1_block_pred * img1_std + img1_mean) * mask_block
                        img2_block_pred = (img2_block_pred * img2_std + img2_mean) * mask_block
                        
                        img_block_pred = (img1_block_pred + img2_block_pred) / 2.0

                        imgavg, vol_count = qtlib.block2brain(img_block_pred, ind_block, mask)

                        fpPred = os.path.join(dpPred, 'n2a_iter' + str(jj) + '_lr1e5_ep' + str(mm) + f'_img_ch{ch}.nii.gz')
                        qtlib.save_nii(fpPred, imgavg, fpImg1)
                        
                        eval_time = time.time() - eval_start_time
                        write_log(log_file, f"✅ Evaluation completed in {eval_time:.2f}s, saved to: {fpPred}")
                        print(f"✅ Evaluation completed in {eval_time:.2f}s")
                        
                        # 清理评估变量
                        del img1_block_pred, img2_block_pred, img1_block_pred_flip, img2_block_pred_flip
                        del img_block_pred
                        gc.collect()
                        
                    except Exception as eval_error:
                        write_log(log_file, f"❌ Evaluation failed: {eval_error}")
                        print(f"❌ Evaluation failed: {eval_error}")
                        if "OOM" in str(eval_error) or "out of memory" in str(eval_error).lower():
                            print("💡 Evaluation OOM - consider reducing block size or batch evaluation")
                    
                    print(f"💾 After evaluation (Epoch {mm}):")
                    print_gpu_status()
            
            # 记录迭代完成时间
            iter_time = time.time() - iter_start_time
            write_log(log_file, f"=== Iteration {jj} completed in {iter_time:.2f}s ===")
            
            # 清理模型和显存
            print("🧹 Cleaning up after iteration...")
            del dtnet
            aggressive_memory_cleanup()
            write_log(log_file, f"Memory cleared after iteration {jj}")
            
            print(f"💾 Final GPU status after Iteration {jj}:")
            print_gpu_status()

        write_log(log_file, f"=== Training completed for subject {sj} ===")

print(f"\n🎉 Training completed for subject {args.subject_idx}: {subjects[args.subject_idx]}")
print("="*60)
print("🏁 FINAL GPU STATUS")
print("="*60)
print_gpu_status()
