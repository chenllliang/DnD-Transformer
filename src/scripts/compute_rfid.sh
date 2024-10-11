export CUDA_VISIBLE_DEVICES=0
# python compute_rfid.py --split=val --vqvae=/cpfs01/user/cl424408/rq-vae-transformer-main/save_ckpts_rq256_with_sync/in256-rqvae-8x8x4-withsyndog/08072024_093732/epoch15_model.pt
# python compute_rfid.py --split=val --vqvae=/cpfs01/user/cl424408/rq-vae-transformer-main/trained_models/in512_32x32x1_imagenet_shared_bsz20*8/in512-rqvae-32x32x1-arxiv-withaux/17092024_105736/epoch25_model.pt
python compute_rfid.py --split=val --vqvae=/cpfs01/user/cl424408/rq-vae-transformer-main/trained_models/in512_32x32x3_imagenet_shared_bsz20*8/in512-rqvae-32x32x3-arxiv-withaux/20092024_071702/epoch75_model.pt
python compute_rfid.py --split=val --vqvae=/cpfs01/user/cl424408/rq-vae-transformer-main/trained_models/in512_32x32x4_imagenet_shared_bsz20*8/in512-rqvae-32x32x4-arxiv-withaux/20092024_071729/epoch30_model.pt
python compute_rfid.py --split=val --vqvae=/cpfs01/user/cl424408/rq-vae-transformer-main/trained_models/in512_32x32x4_imagenet_shared_bsz20*8/in512-rqvae-32x32x4-arxiv-withaux/20092024_071729/epoch40_model.pt
python compute_rfid.py --split=val --vqvae=/cpfs01/user/cl424408/rq-vae-transformer-main/trained_models/in512_32x32x4_imagenet_shared_bsz20*8/in512-rqvae-32x32x4-arxiv-withaux/20092024_071729/epoch50_model.pt
python compute_rfid.py --split=val --vqvae=/cpfs01/user/cl424408/rq-vae-transformer-main/trained_models/in512_32x32x4_imagenet_shared_bsz20*8/in512-rqvae-32x32x4-arxiv-withaux/20092024_071729/epoch60_model.pt
python compute_rfid.py --split=val --vqvae=/cpfs01/user/cl424408/rq-vae-transformer-main/trained_models/in512_32x32x4_imagenet_shared_bsz20*8/in512-rqvae-32x32x4-arxiv-withaux/20092024_071729/epoch70_model.pt
python compute_rfid.py --split=val --vqvae=/cpfs01/user/cl424408/rq-vae-transformer-main/trained_models/in512_32x32x4_imagenet_shared_bsz20*8/in512-rqvae-32x32x4-arxiv-withaux/20092024_071729/epoch74_model.pt




