import gradio as gr
import os, subprocess, yaml

class WebUI:
    def __init__(self) -> None:
        self.info = Info()
        self.opt_cfg_pth = 'configs/opt.yaml'
        self.main_ui()
    
    def main_ui(self):
        with gr.Blocks() as ui:
            gr.Markdown('훈련 및 추론을 위한 편리한 DDSP 웹UI입니다. 각 단계의 설명은 아래에서 확인할 수 있습니다.')
            with gr.Tab("훈련/Training"):
                gr.Markdown(self.info.general)
                with gr.Accordion('사전 훈련 모델 설명', open=False):
                    gr.Markdown(self.info.pretrain_model)
                with gr.Accordion('데이터셋 설명', open=False):
                    gr.Markdown(self.info.dataset)
                
                gr.Markdown('## 설정 파일 생성')
                with gr.Row():
                    self.batch_size = gr.Slider(minimum=2, maximum=60, value=24, label='배치 크기', interactive=True)
                    self.learning_rate = gr.Number(value=0.0005, label='학습률', info='배치 크기와 관련이 약 0.0001:6입니다.')
                    self.f0_extractor = gr.Dropdown(['parselmouth', 'dio', 'harvest', 'crepe'], type='value', value='crepe', label='f0 추출기 유형', interactive=True)
                    self.sampling_rate = gr.Number(value=44100, label='샘플링률', info='데이터셋 오디오의 샘플링률', interactive=True)
                    self.n_spk = gr.Number(value=1, label='화자 수', interactive=True)
                with gr.Row():
                    self.device = gr.Dropdown(['cuda','cpu'], value='cuda', label='사용 장치', interactive=True)
                    self.num_workers = gr.Number(value=2, label='데이터 읽기 프로세스 수', info='장치 성능이 좋으면 0으로 설정할 수 있습니다.', interactive=True)
                    self.cache_all_data = gr.Checkbox(value=True, label='캐시 사용', info='데이터를 모두로드하여 훈련을 가속화합니다.', interactive=True)
                    self.cache_device = gr.Dropdown(['cuda','cpu'], value='cuda', type='value', label='캐시 장치', info='GPU 메모리가 충분하면 cuda로 설정하세요.', interactive=True)
                self.bt_create_config = gr.Button(value='설정 파일 생성')
                
                gr.Markdown('전처리')
                with gr.Accordion('사전 훈련 설명', open=False):
                    gr.Markdown(self.info.preprocess)
                with gr.Row():
                    self.bt_open_data_folder = gr.Button('데이터셋 폴더 열기')
                    self.bt_preprocess = gr.Button('전처리 시작')
                gr.Markdown('훈련')
                with gr.Accordion('훈련 설명', open=False):
                    gr.Markdown(self.info.train)
                with gr.Row():
                    self.bt_train = gr.Button('훈련 시작')
                    self.bt_visual = gr.Button('시각화 시작')
                    gr.Markdown('시각화 시작 후 [여기를 클릭](http://127.0.0.1:6006)')
                
            with gr.Tab('추론/Inference'):
                with gr.Accordion('추론 설명', open=False):
                    gr.Markdown(self.info.infer)
                with gr.Row():
                    self.input_wav = gr.Audio(type='filepath', label='변환할 오디오 선택')
                    self.choose_model = gr.Textbox('exp/model_chino.pt', label='모델 경로')
                with gr.Row():
                    self.keychange = gr.Slider(-24,24,value=0,step=1,label='음정 변경')
                    self.id = gr.Number(value=1,label='화자 ID')
                    self.enhancer_adaptive_key = gr.Number(value=0,label='강화기 음역 오프셋',info='고음(예: G5 이상)을 방지하는 데 도움이 됩니다. 그러나 낮은 소리 품질이 약간 저하될 수 있습니다.')
                with gr.Row():
                    self.bt_infer = gr.Button(value='변환 시작')
                    self.output_wav = gr.Audio(type='filepath',label='출력 오디오')
                    
            self.bt_create_config.click(fn=self.create_config)
            self.bt_open_data_folder.click(fn=self.openfolder)
            self.bt_preprocess.click(fn=self.preprocess)
            self.bt_train.click(fn=self.training)
            self.bt_visual.click(fn=self.visualize)
            self.bt_infer.click(fn=self.inference,inputs=[self.input_wav,self.choose_model,self.keychange,self.id,self.enhancer_adaptive_key],outputs=self.output_wav)
        ui.launch(inbrowser=True,server_port=7858)
        
    def openfolder(self):
        try:
            os.startfile('data')
        except:
            print('폴더 열기 실패!')


    def create_config(self):
        with open('configs/combsub.yaml','r',encoding='utf-8') as f:
            cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
        cfg['data']['f0_extractor'] = str(self.f0_extractor.value)
        cfg['data']['sampling_rate'] = int(self.sampling_rate.value)
        cfg['train']['batch_size'] = int(self.batch_size.value)
        cfg['device'] = str(self.device.value)
        cfg['train']['num_workers'] = int(self.num_workers.value)
        cfg['train']['cache_all_data'] = str(self.cache_all_data.value)
        cfg['train']['cache_device'] = str(self.cache_device.value)
        cfg['train']['lr'] = int(self.learning_rate.value)
        print('설정 파일 정보:'+str(cfg))
        with open(self.opt_cfg_pth,'w',encoding='utf-8') as f:
            yaml.dump(cfg,f)
        print('설정 파일 생성 완료')

    
    def preprocess(self):
        전처리_프로세스 = subprocess.Popen('python -u preprocess.py -c '+self.opt_cfg_pth, stdout=subprocess.PIPE)
        while 전처리_프로세스.poll() is None:
            출력 = 전처리_프로세스.stdout.readline().decode('utf-8')
            print(output)
        print('전처리 완료')
            
    def training(self):
        train_process=subprocess.Popen('python -u train.py -c '+self.opt_cfg_pth,stdout=subprocess.PIPE)
        while train_process.poll() is None:
            output=train_process.stdout.readline().decode('utf-8')
            print(output)
        
            
    def visualize(self):
        tb_process=subprocess.Popen('tensorboard --logdir=exp --port=6006',stdout=subprocess.PIPE)
        while tb_process.poll() is None:
            output=tb_process.stdout.readline().decode('utf-8')
            print(output)
            
    def inference(self,input_wav:str,model:str,keychange,id,enhancer_adaptive_key):
        print(input_wav,model)
        output_wav='samples/'+ input_wav.replace('\\','/').split('/')[-1]
        cmd='python -u main.py -i '+input_wav+' -m '+model+' -o '+output_wav+' -k '+str(int(keychange))+' -id '+str(int(id))+' -e true -eak '+str(int(enhancer_adaptive_key))
        infer_process=subprocess.Popen(cmd,stdout=subprocess.PIPE)
        while infer_process.poll() is None:
            output=infer_process.stdout.readline().decode('utf-8')
            print(output)
        print('추론완료')
        return output_wav







webui=WebUI()