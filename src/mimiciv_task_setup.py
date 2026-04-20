import torch
import numpy as np

from src.crossattnperceiver import InputModality
from src.datasets.embed.get_data_embed import data_prepare as prepare_embed
from src.datasets.mimic.get_data_mimic_iv import data_prepare as prepare_mimic
from src.encoders import EMBEDEncoder, FSEncoder, ModalityEncoders
from src.get_data_eicu import data_prepare as prepare_eicu
from src.shared_encoders import TimeQueryEncoder
from src.loss import FocalLoss

def setup_tasks_and_modalities(args, device, tokenizer, modeltype, modalities, BioBert):
    tasks = args.task.split("-")
    all_train = []
    all_valid = []
    all_test = []
    criterion = []
    modalities_per_task = []
    train_weights = []
    all_encoders = {}
    logits = torch.nn.ModuleList()

    if 'ihm' in tasks:
        # import pdb; pdb.set_trace()
        train_ihm, valid_ihm, test_ihm = prepare_mimic(args=args, task='ihm', tokenizer=tokenizer, modeltype=modeltype['ihm'])
        all_train.append(train_ihm)
        all_valid.append(valid_ihm)
        all_test.append(test_ihm)
        criterion.append(torch.nn.CrossEntropyLoss())
        train_weights.append(1.0)
        ihm_mods = list(map(lambda s: s + '_IHM', args.ihm_mod.split("-")))
        assert len(ihm_mods) > 1, "At least two modalities per task!"
        modalities_per_task.append(ihm_mods)
        if args.fusion_model in ['fusemoe', 'flexmoe']:
            logit_dim = len(ihm_mods) * args.embed_dim
        else:
            logit_dim = len(ihm_mods) * (len(ihm_mods) - 1) * args.perceiver_dim
        logits.append(torch.nn.Sequential(torch.nn.LayerNorm(logit_dim), torch.nn.Linear(logit_dim, 2)))
        if args.shared_modality_encoders:
            shared_time_encoder = TimeQueryEncoder(
                tt_max=args.tt_max,
                embed_time=args.embed_time,
                device=device
            )
        else:
            shared_time_encoder = None
        ihm_encoder = ModalityEncoders(
            args,
            device,
            modalities,
            args.tt_max,
            args.num_of_notes,
            BioBert,
            shared_time_encoder=shared_time_encoder.to(device)
        )
        all_encoders['IHM'] = ihm_encoder

    if 'los' in tasks:
        train_los, valid_los, test_los = prepare_mimic(args=args, task='los', tokenizer=tokenizer, modeltype=modeltype['los'])
        all_train.append(train_los)
        all_valid.append(valid_los)
        all_test.append(test_los)
        criterion.append(torch.nn.CrossEntropyLoss())
        train_weights.append(1.0)
        los_mods = list(map(lambda s: s + '_LOS', args.los_mod.split("-")))
        assert len(los_mods) > 1, "At least two modalities per task!"
        modalities_per_task.append(los_mods)
        if args.fusion_model in ['fusemoe', 'flexmoe']:
            logit_dim = len(los_mods) * args.embed_dim
        else:
            logit_dim = len(los_mods) * (len(los_mods) - 1) * args.perceiver_dim
        logits.append(torch.nn.Sequential(torch.nn.LayerNorm(logit_dim), torch.nn.Linear(logit_dim, 2)))

        if 'IHM' in all_encoders and args.shared_modality_encoders:
            all_encoders['LOS'] = all_encoders['IHM']
        else:
            if args.shared_modality_encoders:
                shared_time_encoder = TimeQueryEncoder(
                tt_max=args.tt_max,
                embed_time=args.embed_time,
                device=device
            )
            else:
                shared_time_encoder = None
            los_encoder = ModalityEncoders(
                args,
                device,
                modalities,
                args.tt_max,
                args.num_of_notes,
                BioBert,
                shared_time_encoder=shared_time_encoder.to(device)
            )
            all_encoders['LOS'] = los_encoder

    if 'pheno' in tasks:
        train_pheno, valid_pheno, test_pheno = prepare_mimic(args=args, task='pheno', tokenizer=tokenizer, modeltype=modeltype['pheno'])
        all_train.append(train_pheno)
        all_valid.append(valid_pheno)
        all_test.append(test_pheno)
        criterion.append(torch.nn.BCEWithLogitsLoss())
        train_weights.append(1.0)
        pheno_mods = list(map(lambda s: s + '_PHENO', args.pheno_mod.split("-")))
        assert len(pheno_mods) > 1, "At least two modalities per task!"
        modalities_per_task.append(pheno_mods)
        if args.fusion_model in ['fusemoe', 'flexmoe']:
            logit_dim = len(pheno_mods) * args.embed_dim
        else:
            logit_dim = len(pheno_mods) * (len(pheno_mods) - 1) * args.perceiver_dim
        logits.append(torch.nn.Sequential(torch.nn.LayerNorm(logit_dim), torch.nn.Linear(logit_dim, 25)))

        if 'IHM' in all_encoders:
            all_encoders['PHENO'] = all_encoders['IHM']
        elif 'LOS' in all_encoders:
            all_encoders['PHENO'] = all_encoders['LOS']
        else:
            if args.shared_modality_encoders:
                    shared_time_encoder = TimeQueryEncoder(
                    tt_max=args.tt_max,
                    embed_time=args.embed_time,
                    device=device
                )
            else:
                shared_time_encoder = None

            pheno_encoder = ModalityEncoders(
                args,
                device,
                modalities,
                args.tt_max,
                args.num_of_notes,
                BioBert,
                shared_time_encoder=shared_time_encoder.to(device)
            )
            all_encoders['PHENO'] = pheno_encoder

    if 'readmission' in tasks:
        train_rad, valid_rad, test_rad, tokenizer_rad = prepare_eicu(args=args)
        all_train.append(train_rad)
        all_valid.append(valid_rad)
        all_test.append(test_rad)
        criterion.append(torch.nn.CrossEntropyLoss())
        train_weights.append(1.0)
        rad_mods = list(map(lambda s: s + '_RAD', args.rad_mod.split("-")))
        modalities_per_task.append(rad_mods)
        if args.fusion_model in ['fusemoe', 'flexmoe']:
            logit_dim = len(rad_mods) * args.embed_dim
        else:
            logit_dim = len(rad_mods) * (len(rad_mods) - 1) * args.perceiver_dim
        logits.append(torch.nn.Sequential(torch.nn.LayerNorm(logit_dim), torch.nn.Linear(logit_dim, 2)))
        if args.shared_modality_encoders:
            shared_time_encoder = TimeQueryEncoder(
                tt_max=args.tt_max,
                embed_time=args.embed_time,
                device=device
            )
        else:
            shared_time_encoder = None
        readmission_encoder = FSEncoder(
            tokenizer=tokenizer_rad,
            embedding_size=args.embed_dim,
            pretrained_embedding=args.use_pt_text_embeddings,
            dropout=args.dropout,
            layers=args.layers,
            heads=args.num_heads,
            hidden_size=args.hidden_size,
            device=device,
            shared_time_encoder=shared_time_encoder.to(device),
            args=args
        )
        all_encoders['RAD'] = readmission_encoder

    if 'mortality' in tasks:
        train_mor, valid_mor, test_mor, tokenizer_mor = prepare_eicu(args=args)
        all_train.append(train_mor)
        all_valid.append(valid_mor)
        all_test.append(test_mor)
        criterion.append(torch.nn.CrossEntropyLoss())
        train_weights.append(1.0)
        mor_mods = list(map(lambda s: s + '_MOR', args.mor_mod.split("-")))
        modalities_per_task.append(mor_mods)
        if args.fusion_model in ['fusemoe', 'flexmoe']:
            logit_dim = len(mor_mods) * args.embed_dim
        else:
            logit_dim = len(mor_mods) * (len(mor_mods) - 1) * args.perceiver_dim
        logits.append(torch.nn.Sequential(torch.nn.LayerNorm(logit_dim), torch.nn.Linear(logit_dim, 2)))

        if 'RAD' in all_encoders:
            all_encoders['MOR'] = all_encoders['RAD']
        else:
            if args.shared_modality_encoders:
                shared_time_encoder = TimeQueryEncoder(
                    tt_max=args.tt_max,
                    embed_time=args.embed_time,
                    device=device
                )
            else:
                shared_time_encoder = None
            mortality_encoder = FSEncoder(
                tokenizer=tokenizer_mor,
                embedding_size=args.embed_dim,
                pretrained_embedding=args.use_pt_text_embeddings,
                dropout=args.dropout,
                layers=args.layers,
                heads=args.num_heads,
                hidden_size=args.hidden_size,
                device=device,
                shared_time_encoder=shared_time_encoder.to(device),
                args=args
            )
            all_encoders['MOR'] = mortality_encoder
    if 'birads' in tasks:
        train_birads, valid_birads, test_birads, train_dataset, _, _ = prepare_embed(args=args, task='birads', modeltype=modeltype['birads'])
        all_train.append(train_birads)
        all_valid.append(valid_birads)
        all_test.append(test_birads)
        class_dist = np.bincount([d['label'] for d in train_dataset]).astype(float)
        class_weights = torch.tensor(1 - class_dist / class_dist.sum()).float()
        criterion.append(FocalLoss(gamma=2, alpha=class_weights))
        train_weights.append(1.0)
        birads_mods = list(map(lambda s: s + '_BIRADS', args.birads_mod.split("-")))
        modalities_per_task.append(birads_mods)
        if args.fusion_model in ['fusemoe', 'flexmoe']:
            logit_dim = len(birads_mods) * args.embed_dim
        else:
            logit_dim = len(birads_mods) * (len(birads_mods) - 1) * args.perceiver_dim
        logits.append(torch.nn.Sequential(torch.nn.LayerNorm(logit_dim), torch.nn.Linear(logit_dim, 3)))
        if args.shared_modality_encoders:
            shared_time_encoder = TimeQueryEncoder(
                tt_max=args.tt_max,
                embed_time=args.embed_time,
                device=device
            )
        else:
            shared_time_encoder = None
        birads_encoder = EMBEDEncoder(args=args, device=device,modalities=modalities)
        all_encoders['BIRADS'] = birads_encoder
    if 'risk' in tasks:
        train_risk, valid_risk, test_risk, train_dataset, _, _ = prepare_embed(args=args, task='risk', modeltype=modeltype['risk'])
        all_train.append(train_risk)
        all_valid.append(valid_risk)
        all_test.append(test_risk)
        class_dist = np.bincount([d['label'] for d in train_dataset]).astype(float)
        class_weights = torch.tensor(1 - class_dist / class_dist.sum()).float()
        criterion.append(FocalLoss(gamma=2, alpha=class_weights))
        train_weights.append(1.0)
        risk_mods = list(map(lambda s: s + '_RISK', args.risk_mod.split("-")))
        modalities_per_task.append(risk_mods)
        if args.fusion_model in ['fusemoe', 'flexmoe']:
            logit_dim = len(risk_mods) * args.embed_dim
        else:
            logit_dim = len(risk_mods) * (len(risk_mods) - 1) * args.perceiver_dim
        logits.append(torch.nn.Sequential(torch.nn.LayerNorm(logit_dim), torch.nn.Linear(logit_dim, 2)))
        if args.shared_modality_encoders:
            shared_time_encoder = TimeQueryEncoder(
                tt_max=args.tt_max,
                embed_time=args.embed_time,
                device=device
            )
        else:
            shared_time_encoder = None
        risk_encoder = EMBEDEncoder(args=args, device=device, modalities=modalities)
        all_encoders['RISK'] = risk_encoder
    if 'density' in tasks:
        train_density, valid_density, test_density, train_dataset, _, _ = prepare_embed(args=args, task='density', modeltype=modeltype['density'])
        all_train.append(train_density)
        all_valid.append(valid_density)
        all_test.append(test_density)
        class_dist = np.bincount([d['label'] for d in train_dataset]).astype(float)
        class_weights = torch.tensor(1 - class_dist / class_dist.sum()).float()
        criterion.append(FocalLoss(gamma=2, alpha=class_weights))
        train_weights.append(1.0)
        density_mods = list(map(lambda s: s + '_DENSITY', args.density_mod.split("-")))
        modalities_per_task.append(density_mods)
        if args.fusion_model in ['fusemoe', 'flexmoe']:
            logit_dim = len(density_mods) * args.embed_dim
        else:
            logit_dim = len(density_mods) * (len(density_mods) - 1) * args.perceiver_dim
        logits.append(torch.nn.Sequential(torch.nn.LayerNorm(logit_dim), torch.nn.Linear(logit_dim, 4)))
        if args.shared_modality_encoders:
            shared_time_encoder = TimeQueryEncoder(
                tt_max=args.tt_max,
                embed_time=args.embed_time,
                device=device
            )
        else:
            shared_time_encoder = None
        density_encoder = EMBEDEncoder(args=args, device=device, modalities=modalities)
        all_encoders['DENSITY'] = density_encoder

    all_modalities = {}
    if args.shared_modality_encoders:
        all_modalities['Text'] = InputModality(
            name='Text',
            input_channels=args.embed_dim,
            input_axis=1,
            num_freq_bands=6,
            max_freq=1
        )
        all_modalities['TS'] = InputModality(
            name='TS',
            input_channels=args.embed_dim,
            input_axis=1,
            num_freq_bands=6,
            max_freq=1
        )
        all_modalities['CXR'] = InputModality(
            name='CXR',
            input_channels=args.embed_dim,
            input_axis=1,
            num_freq_bands=6,
            max_freq=1
        )
        all_modalities['ECG'] = InputModality(
            name='ECG',
            input_channels=args.embed_dim,
            input_axis=1,
            num_freq_bands=6,
            max_freq=1
        )
        all_modalities['T1'] = InputModality(
            name='T1',
            input_channels=args.embed_dim,
            input_axis=1,
            num_freq_bands=6,
            max_freq=1
        )
        all_modalities['T2'] = InputModality(
            name='T2',
            input_channels=args.embed_dim,
            input_axis=1,
            num_freq_bands=6,
            max_freq=1
        )
        all_modalities['T3'] = InputModality(
            name='T3',
            input_channels=args.embed_dim,
            input_axis=1,
            num_freq_bands=6,
            max_freq=1
        )
        all_modalities['T4'] = InputModality(
            name='T4',
            input_channels=args.embed_dim,
            input_axis=1,
            num_freq_bands=6,
            max_freq=1
        )
        all_modalities['T5'] = InputModality(
            name='T5',
            input_channels=args.embed_dim,
            input_axis=1,
            num_freq_bands=6,
            max_freq=1
        )
        all_modalities['eicu'] = InputModality(
            name='eicu',
            input_channels=args.embed_dim,
            input_axis=1,
            num_freq_bands=6,
            max_freq=1
        )
        all_modalities['cc'] = InputModality(
            name='cc',
            input_channels=args.embed_dim,
            input_axis=1,
            num_freq_bands=6,
            max_freq=1
        )
        all_modalities['mlo'] = InputModality(
            name='mlo',
            input_channels=args.embed_dim,
            input_axis=1,
            num_freq_bands=6,
            max_freq=1
        )
        all_modalities['2dcc'] = InputModality(
            name='2dcc',
            input_channels=args.embed_dim,
            input_axis=1,
            num_freq_bands=6,
            max_freq=1
        )
        all_modalities['2dmlo'] = InputModality(
            name='2dmlo',
            input_channels=args.embed_dim,
            input_axis=1,
            num_freq_bands=6,
            max_freq=1
        )
    else:
        all_modalities['Text_IHM'] = InputModality(
            name='Text_IHM',
            input_channels=args.embed_dim,
            input_axis=1,
            num_freq_bands=6,
            max_freq=1
        )
        all_modalities['TS_IHM'] = InputModality(
            name='TS_IHM',
            input_channels=args.embed_dim,
            input_axis=1,
            num_freq_bands=6,
            max_freq=1
        )
        all_modalities['CXR_IHM'] = InputModality(
            name='CXR_IHM',
            input_channels=args.embed_dim,
            input_axis=1,
            num_freq_bands=6,
            max_freq=1.
        )
        all_modalities['ECG_IHM'] = InputModality(
            name='ECG_IHM',
            input_channels=args.embed_dim,
            input_axis=1,
            num_freq_bands=6,
            max_freq=1
        )
        all_modalities['Text_LOS'] = InputModality(
            name='Text_LOS',
            input_channels=args.embed_dim,
            input_axis=1,
            num_freq_bands=6,
            max_freq=1
        )
        all_modalities['TS_LOS'] = InputModality(
            name='TS_LOS',
            input_channels=args.embed_dim,
            input_axis=1,
            num_freq_bands=6,
            max_freq=1
        )
        all_modalities['CXR_LOS'] = InputModality(
            name='CXR_LOS',
            input_channels=args.embed_dim,
            input_axis=1,
            num_freq_bands=6,
            max_freq=1.
        )
        all_modalities['ECG_LOS'] = InputModality(
            name='ECG_LOS',
            input_channels=args.embed_dim,
            input_axis=1,
            num_freq_bands=6,
            max_freq=1.
        )
        all_modalities['Text_PHENO'] = InputModality(
            name='Text_PHENO',
            input_channels=args.embed_dim,
            input_axis=1,
            num_freq_bands=6,
            max_freq=1
        )
        all_modalities['TS_PHENO'] = InputModality(
            name='TS_PHENO',
            input_channels=args.embed_dim,
            input_axis=1,
            num_freq_bands=6,
            max_freq=1
        )
        all_modalities['CXR_PHENO'] = InputModality(
            name='CXR_PHENO',
            input_channels=args.embed_dim,
            input_axis=1,
            num_freq_bands=6,
            max_freq=1.
        )
        all_modalities['ECG_PHENO'] = InputModality(
            name='ECG_PHENO',
            input_channels=args.embed_dim,
            input_axis=1,
            num_freq_bands=6,
            max_freq=1.
        )
        all_modalities['T1_MOR'] = InputModality(
            name='T1_MOR',
            input_channels=args.embed_dim,
            input_axis=1,
            num_freq_bands=6,
            max_freq=1
        )
        all_modalities['T2_MOR'] = InputModality(
            name='T2_MOR',
            input_channels=args.embed_dim,
            input_axis=1,
            num_freq_bands=6,
            max_freq=1
        )
        all_modalities['T3_MOR'] = InputModality(
            name='T3_MOR',
            input_channels=args.embed_dim,
            input_axis=1,
            num_freq_bands=6,
            max_freq=1
        )
        all_modalities['T4_MOR'] = InputModality(
            name='T4_MOR',
            input_channels=args.embed_dim,
            input_axis=1,
            num_freq_bands=6,
            max_freq=1
        )
        all_modalities['T5_MOR'] = InputModality(
            name='T5_MOR',
            input_channels=args.embed_dim,
            input_axis=1,
            num_freq_bands=6,
            max_freq=1
        )
        all_modalities['T1_RAD'] = InputModality(
            name='T1_RAD',
            input_channels=args.embed_dim,
            input_axis=1,
            num_freq_bands=6,
            max_freq=1
        )
        all_modalities['T2_RAD'] = InputModality(
            name='T2_RAD',
            input_channels=args.embed_dim,
            input_axis=1,
            num_freq_bands=6,
            max_freq=1
        )
        all_modalities['T3_RAD'] = InputModality(
            name='T3_RAD',
            input_channels=args.embed_dim,
            input_axis=1,
            num_freq_bands=6,
            max_freq=1
        )
        all_modalities['T4_RAD'] = InputModality(
            name='T4_RAD',
            input_channels=args.embed_dim,
            input_axis=1,
            num_freq_bands=6,
            max_freq=1
        )
        all_modalities['T5_RAD'] = InputModality(
            name='T5_RAD',
            input_channels=args.embed_dim,
            input_axis=1,
            num_freq_bands=6,
            max_freq=1
        )

    return (
        all_train,
        all_valid,
        all_test,
        criterion,
        modalities_per_task,
        train_weights,
        all_encoders,
        logits,
        all_modalities,
    )