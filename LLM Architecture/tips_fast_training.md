Ideas for speeding up NN training and hyperparameter search:

Key levers, roughly ordered by impact:

  DataLoader workers — yes, set num_workers=4 (or match CPU core count). Also add pin_memory=True when using  
  GPU — moves data to pinned RAM so GPU transfer is faster:
  DataLoader(..., num_workers=4, pin_memory=True)                                                             
                                                                             
  Mixed precision (AMP) — biggest single win after GPU. Cuts memory ~50%, speeds up training 2-3x on modern
  GPUs. Minimal code change:                                                                                  
  from torch.cuda.amp import autocast, GradScaler
  scaler = GradScaler()                                                                                       
                                                                             
  # in train loop:                             
  with autocast():                                                                                            
      y_pred = model(X.to(device))
      loss = criterion(y_pred, y.to(device))                                                                  
  scaler.scale(loss).backward()                                              
  scaler.step(optimizer)                       
  scaler.update()                                                                                             
                 
  Batch size — bigger batch = fewer optimizer steps per epoch = faster. Double it until GPU memory runs out.  
  Watch that accuracy doesn't degrade — may need to scale LR up proportionally (lr * batch_size /             
  original_batch_size).                        
                                                                                                              
  Reduce epochs for bad configs — in hyperparameter search, kill runs early if validation loss isn't improving
   after N epochs. No point training 60 epochs on a bad LR. Simple manual early stopping:
  if test_loss > best_loss * 1.05:  # 5% worse than best                                                      
      break                                                                  
                                                                                                              
  Parallel search — if GPU has headroom, run multiple small models simultaneously using Python multiprocessing
   or just split the search space and run two terminal sessions. Only works if individual model fits in <50%  
  GPU memory.                                                                
                                                                                                              
  torch.compile (PyTorch 2.x) — one line, ~10-30% speedup:                                                    
  net = torch.compile(net)                     
                                                                                                              
  Priority order for your case:                                              
  1. num_workers=4, pin_memory=True — free win, 5 min to add
  2. Mixed precision — biggest speedup, ~30 min to add                                                        
  3. Early stopping — cuts wasted time on bad configs 
  4. Bigger batch size — tune carefully                                                                       
  5. torch.compile — last, easiest but least predictable gain   


---                                                                                                         
  torch.compile — nothing to do with multiprocessing. It's a JIT compiler — takes your model's computation
  graph, optimizes it (fuses operations, eliminates redundant work), produces faster native code. One line,   
  wraps existing model:                                                                                    
  net = Net().to(device)                                                                                      
  net = torch.compile(net)  # same object, now compiled                      
  First forward pass is slow (compilation happens). All subsequent passes faster.
                                                                                                              
  ---                                                                                                         
  Loss rising while accuracy improves — real phenomenon, not imagination. Happens when model becomes more     
  confident on correct predictions but also more confident on wrong ones. Loss penalizes confidence of errors 
  heavily (log scale), so overconfident wrong predictions spike loss. Accuracy only counts right/wrong,      
  ignores confidence. Common near end of training when model starts overfitting. Means: accuracy is lagging   
  indicator here, loss caught overfitting first.                                                           
                                                
  Implication for early stopping: monitor accuracy, not loss, if accuracy is your goal.
                                                                                                              
  ---
  pin_memory=True — CPU RAM has two types: pageable (normal) and pinned (page-locked). GPU can only           
  DMA-transfer from pinned memory directly. Normal flow: CPU pageable → CPU pinned → GPU. With pin_memory=True
   DataLoader allocates batches directly in pinned RAM, skipping first copy. Faster data-to-GPU transfer,
  especially noticeable with large batches. Not caching — just skips one memory copy per batch.               
                                                                             
  ---                                          
  Epochs and accuracy — yes, fewer epochs = lower accuracy for good configs. Early stopping should only kill
  bad configs early, not cap good ones. Pattern: set a minimum epoch threshold (e.g. 10) before any killing,  
  then stop if no improvement for N epochs. Good configs survive, bad ones die fast.