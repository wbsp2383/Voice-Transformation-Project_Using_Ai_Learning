import os
import time
import numpy as np
import torch

from logger.saver import Saver
from logger import utils

def test(args, model, loss_func, loader_test, saver):
    print(' [*] testing...')
    model.eval()

   
    test_loss = 0.
    test_loss_rss = 0.
    test_loss_uv = 0.
    
    
    num_batches = len(loader_test)
    rtf_all = []
    
    # 구동
    with torch.no_grad():
        for bidx, data in enumerate(loader_test):
            fn = data['name'][0]
            print('--------')
            print('{}/{} - {}'.format(bidx, num_batches, fn))

            # 데이터 해제
            for k in data.keys():
                if k != 'name':
                    data[k] = data[k].to(args.device)
            print('>>', data['name'][0])

            # 출력생성
            st_time = time.time()
            signal, _, (s_h, s_n) = model(data['units'], data['f0'], data['volume'], data['spk_id'])
            ed_time = time.time()

            # 음원 잘라내기
            min_len = np.min([signal.shape[1], data['audio'].shape[1]])
            signal        = signal[:,:min_len]
            data['audio'] = data['audio'][:,:min_len]

            # 텍스트 형식 보존
            run_time = ed_time - st_time
            song_time = data['audio'].shape[-1] / args.data.sampling_rate
            rtf = run_time / song_time
            print('RTF: {}  | {} / {}'.format(rtf, run_time, song_time))
            rtf_all.append(rtf)
           
            
            loss = loss_func(signal, data['audio'])

            test_loss += loss.item()

            
            saver.log_audio({fn+'/gt.wav': data['audio'], fn+'/pred.wav': signal})
            
    # 기록
    test_loss /= num_batches
    
    # 확인
    print(' [test_loss] test_loss:', test_loss)
    print(' Real Time Factor', np.mean(rtf_all))
    return test_loss


def train(args, initial_global_step, model, optimizer, loss_func, loader_train, loader_test):
    # 데이터 저장
    saver = Saver(args, initial_global_step=initial_global_step)

    # 대상 크기
    params_count = utils.get_network_paras_amount({'model': model})
    saver.log_info('--- model size ---')
    saver.log_info(params_count)
    
    # 시행
    num_batches = len(loader_train)
    model.train()
    saver.log_info('======= start training =======')
    for epoch in range(args.train.epochs):
        for batch_idx, data in enumerate(loader_train):
            saver.global_step_increment()
            optimizer.zero_grad()

            # 데이터 언팩
            for k in data.keys():
                if k != 'name':
                    data[k] = data[k].to(args.device)
            
            
            signal, _, (s_h, s_n) = model(data['units'].float(), data['f0'], data['volume'], data['spk_id'], infer=False)

            
            loss = loss_func(signal, data['audio'])
            
            # 손실처리
            if torch.isnan(loss):
                raise ValueError(' [x] nan loss ')
            else:
                # 함수 기울기 계산
                loss.backward()
                optimizer.step()

            # 로그손실
            if saver.global_step % args.train.interval_log == 0:
                saver.log_info(
                    'epoch: {} | {:3d}/{:3d} | {} | batch/s: {:.2f} | loss: {:.3f} | time: {} | step: {}'.format(
                        epoch,
                        batch_idx,
                        num_batches,
                        args.env.expdir,
                        args.train.interval_log/saver.get_interval_time(),
                        loss.item(),
                        saver.get_total_time(),
                        saver.global_step
                    )
                )
                
                saver.log_value({
                    'train/loss': loss.item()
                })
            
            # 검증
            if saver.global_step % args.train.interval_val == 0:
                optimizer_save = optimizer if args.train.save_opt else None
                    
                
                saver.save_model(model, optimizer_save, postfix=f'{saver.global_step}')

                
                test_loss = test(args, model, loss_func, loader_test, saver)
                
                
                saver.log_info(
                    ' --- <validation> --- \nloss: {:.3f}. '.format(
                        test_loss,
                    )
                )
    
                saver.log_value({
                    'validation/loss': test_loss
                })
                
                model.train()

                          
