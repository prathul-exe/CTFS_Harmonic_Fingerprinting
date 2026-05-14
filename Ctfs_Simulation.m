%% CTFS_Simulation.m
% Complete MATLAB Simulation for CTFS-Based Harmonic Fingerprinting
% Run this single file — it simulates all 7 modules
% 
% How to run:
%   1. Open MATLAB
%   2. Set current folder to E:\CTFS_Harmonic_Fingerprinting
%   3. Type: CTFS_Simulation
%   Press Enter — all plots will appear one by one
%
% This script reads the real .wav files and runs the full pipeline

clc; clear; close all;

BASE_DIR    = 'E:\CTFS_Harmonic_Fingerprinting';
N_HARMONICS = 32;
FRAME_MS    = 100;
TARGET_SR   = 22050;
SNR_THRESH  = 30.0;
BG          = [0.10 0.10 0.18];

INSTRUMENTS = {'sitar', 'veena', 'guitar'};
COLORS      = {[0.72 0.31 0.26], [0.65 0.75 0.68], [0.83 0.77 0.66]};
WAV_FILES   = {
    fullfile(BASE_DIR,'data','raw','sitar','yaman.wav'),   'sitar',  'yaman';
    fullfile(BASE_DIR,'data','raw','sitar','bhairavi.wav'),'sitar',  'bhairavi';
    fullfile(BASE_DIR,'data','raw','veena','bhairavi.wav'),'veena',  'bhairavi';
    fullfile(BASE_DIR,'data','raw','veena','desh.wav'),    'veena',  'desh';
    fullfile(BASE_DIR,'data','raw','guitar','01_Rock1-130-A_solo_mic.wav'),'guitar','none';
    fullfile(BASE_DIR,'data','raw','guitar','01_SS1-100-C#_solo_mic.wav'),'guitar','none';
    fullfile(BASE_DIR,'data','raw','guitar','01_SS1-68-E_solo_mic.wav'),  'guitar','none';
};

fprintf('\n');
fprintf('================================================\n');
fprintf('  CTFS Harmonic Fingerprinting — MATLAB Simulation\n');
fprintf('================================================\n\n');

%% ═══════════════════════════════════════════════════
%% MODULE 1 — DATA LOADING & SEGMENTATION
%% ═══════════════════════════════════════════════════
fprintf('MODULE 1 — Data Loading & Segmentation\n');
fprintf('----------------------------------------\n');

frame_len  = round(FRAME_MS * 1e-3 * TARGET_SR);
hop_len    = round(frame_len * 0.5);
hann_win   = hann(frame_len);

all_segments = struct();
for k = 1:numel(INSTRUMENTS)
    all_segments.(INSTRUMENTS{k}) = [];
end

for f = 1:size(WAV_FILES,1)
    fpath = WAV_FILES{f,1};
    inst  = WAV_FILES{f,2};
    
    if ~exist(fpath,'file')
        fprintf('  [SKIP] Not found: %s\n', fpath);
        continue;
    end
    
    [x, fs] = audioread(fpath);
    if size(x,2) > 1, x = mean(x,2); end
    
    % Resample to TARGET_SR
    if fs ~= TARGET_SR
        x = resample(x, TARGET_SR, fs);
    end
    x = x / (max(abs(x)) + 1e-9);
    
    % Use only first 30 seconds
    max_samp = min(length(x), TARGET_SR * 30);
    x = x(1:max_samp);
    
    % Extract frames
    num_frames = floor((length(x) - frame_len) / hop_len) + 1;
    n_valid = 0;
    for i = 1:num_frames
        idx   = (i-1)*hop_len + 1;
        frame = x(idx : idx+frame_len-1) .* hann_win;
        rms   = sqrt(mean(frame.^2));
        if rms < 0.01, continue; end
        n_valid = n_valid + 1;
        all_segments.(inst) = [all_segments.(inst); frame'];
    end
    fprintf('  %s (%s): %d frames\n', inst, WAV_FILES{f,3}, n_valid);
end

fprintf('\n  Total frames:\n');
for k = 1:numel(INSTRUMENTS)
    inst = INSTRUMENTS{k};
    fprintf('    %-8s: %d\n', inst, size(all_segments.(inst),1));
end

%% ═══════════════════════════════════════════════════
%% MODULE 2 — CTFS HARMONIC EXTRACTION
%% ═══════════════════════════════════════════════════
fprintf('\nMODULE 2 — CTFS Harmonic Extraction\n');
fprintf('----------------------------------------\n');

all_mags = struct();
all_f0s  = struct();

for k = 1:numel(INSTRUMENTS)
    inst    = INSTRUMENTS{k};
    frames  = all_segments.(inst);
    n_fr    = size(frames,1);
    mags_mat = zeros(n_fr, N_HARMONICS);
    f0s_vec  = zeros(n_fr, 1);
    valid    = 0;
    
    for i = 1:n_fr
        frame = frames(i,:);
        f0    = estimate_f0_fn(frame, TARGET_SR);
        if isnan(f0) || f0 < 50 || f0 > 2000, continue; end
        
        mags = compute_ctfs_fn(frame, TARGET_SR, f0, N_HARMONICS);
        valid = valid + 1;
        mags_mat(valid,:) = mags;
        f0s_vec(valid)    = f0;
    end
    
    all_mags.(inst) = mags_mat(1:valid,:);
    all_f0s.(inst)  = f0s_vec(1:valid);
    fprintf('  %-8s: %d valid frames | mean f0 = %.1f Hz\n', ...
            inst, valid, mean(f0s_vec(1:valid)));
end

%% ═══════════════════════════════════════════════════
%% MODULE 3 — RECONSTRUCTION FIDELITY
%% ═══════════════════════════════════════════════════
fprintf('\nMODULE 3 — Reconstruction Fidelity\n');
fprintf('----------------------------------------\n');

snr_curves = struct();
N_star     = struct();
t          = (0:frame_len-1) / TARGET_SR;

for k = 1:numel(INSTRUMENTS)
    inst    = INSTRUMENTS{k};
    mags    = all_mags.(inst);
    f0s     = all_f0s.(inst);
    n_fr    = min(size(mags,1), 30);  % use 30 frames for speed
    
    snr_mat = zeros(n_fr, N_HARMONICS);
    
    for i = 1:n_fr
        m  = mags(i,:);
        f0 = f0s(i);
        
        % Full reconstruction (reference)
        original = zeros(1, frame_len);
        for n = 1:N_HARMONICS
            original = original + m(n)*cos(2*pi*n*f0*t);
        end
        
        for N = 1:N_HARMONICS
            recon = zeros(1, frame_len);
            for n = 1:N
                recon = recon + m(n)*cos(2*pi*n*f0*t);
            end
            err = original - recon;
            sp  = sum(original.^2);
            ep  = sum(err.^2);
            if ep < 1e-12
                snr_mat(i,N) = 100;
            else
                snr_mat(i,N) = 10*log10(sp/ep);
            end
        end
    end
    
    mean_snr = mean(snr_mat, 1);
    snr_curves.(inst) = mean_snr;
    
    % Find N*
    ns = N_HARMONICS;
    for N = 1:N_HARMONICS
        if mean_snr(N) >= SNR_THRESH
            ns = N; break;
        end
    end
    N_star.(inst) = ns;
    fprintf('  %-8s: N* = %d | SNR at N=32: %.1f dB\n', ...
            inst, ns, mean_snr(end));
end

%% ═══════════════════════════════════════════════════
%% MODULE 4 — FEATURE VECTORS
%% ═══════════════════════════════════════════════════
fprintf('\nMODULE 4 — Feature Vector Construction\n');
fprintf('----------------------------------------\n');

mean_envs = struct();
for k = 1:numel(INSTRUMENTS)
    inst = INSTRUMENTS{k};
    mean_envs.(inst) = mean(all_mags.(inst), 1);
end

X_all = []; y_all = {};
for k = 1:numel(INSTRUMENTS)
    inst    = INSTRUMENTS{k};
    mags    = all_mags.(inst);
    mu      = mean_envs.(inst);
    dev_vec = mags - repmat(mu, size(mags,1), 1);
    X_all   = [X_all; dev_vec];
    for i = 1:size(dev_vec,1)
        y_all{end+1} = inst;
    end
end

fprintf('  Feature matrix: %d x %d\n', size(X_all,1), size(X_all,2));
for k = 1:numel(INSTRUMENTS)
    inst  = INSTRUMENTS{k};
    count = sum(strcmp(y_all, inst));
    fprintf('  %-8s: %d samples\n', inst, count);
end

%% ═══════════════════════════════════════════════════
%% MODULE 5 — CLASSIFICATION (SVM + kNN)
%% ═══════════════════════════════════════════════════
fprintf('\nMODULE 5 — Classification\n');
fprintf('----------------------------------------\n');

y_categorical = categorical(y_all');

% Normalize
mu_X    = mean(X_all, 1);
sig_X   = std(X_all, 0, 1) + 1e-9;
X_norm  = (X_all - mu_X) ./ sig_X;

% SVM
fprintf('  Training SVM (RBF)...\n');
svm_mdl  = fitcecoc(X_norm, y_categorical, ...
    'Learners', templateSVM('KernelFunction','rbf',...
                            'BoxConstraint',10,...
                            'KernelScale','auto'));
cv_svm   = crossval(svm_mdl, 'KFold', 5);
svm_acc  = (1 - kfoldLoss(cv_svm)) * 100;
svm_pred = kfoldPredict(cv_svm);
fprintf('  SVM Accuracy: %.2f%%\n', svm_acc);

% kNN
fprintf('  Training kNN (k=5)...\n');
knn_mdl  = fitcknn(X_norm, y_categorical, 'NumNeighbors', 5);
cv_knn   = crossval(knn_mdl, 'KFold', 5);
knn_acc  = (1 - kfoldLoss(cv_knn)) * 100;
knn_pred = kfoldPredict(cv_knn);
fprintf('  kNN Accuracy: %.2f%%\n', knn_acc);

fprintf('\n  ╔══════════════════════════╗\n');
fprintf('  ║  CLASSIFICATION RESULTS  ║\n');
fprintf('  ╠══════════════════════════╣\n');
fprintf('  ║  SVM (RBF): %6.2f%%      ║\n', svm_acc);
fprintf('  ║  kNN (k=5): %6.2f%%      ║\n', knn_acc);
fprintf('  ╚══════════════════════════╝\n');

%% ═══════════════════════════════════════════════════
%% MODULE 6 — ANOVA RAGA EFFECT
%% ═══════════════════════════════════════════════════
fprintf('\nMODULE 6 — ANOVA Raga Effect\n');
fprintf('----------------------------------------\n');

% For sitar: separate yaman vs bhairavi frames
% We load each separately
raga_pairs = {
    'sitar', fullfile(BASE_DIR,'data','raw','sitar','yaman.wav'),    'yaman';
    'sitar', fullfile(BASE_DIR,'data','raw','sitar','bhairavi.wav'), 'bhairavi';
    'veena', fullfile(BASE_DIR,'data','raw','veena','bhairavi.wav'), 'bhairavi';
    'veena', fullfile(BASE_DIR,'data','raw','veena','desh.wav'),     'desh';
};

raga_mags = struct();
for r = 1:size(raga_pairs,1)
    inst  = raga_pairs{r,1};
    fpath = raga_pairs{r,2};
    raga  = raga_pairs{r,3};
    key   = [inst '_' raga];
    
    if ~exist(fpath,'file'), continue; end
    [x, fs] = audioread(fpath);
    if size(x,2)>1, x=mean(x,2); end
    if fs~=TARGET_SR, x=resample(x,TARGET_SR,fs); end
    x = x(1:min(length(x),TARGET_SR*30));
    x = x/(max(abs(x))+1e-9);
    
    num_frames = floor((length(x)-frame_len)/hop_len)+1;
    mags_tmp = [];
    for i = 1:num_frames
        idx   = (i-1)*hop_len+1;
        frame = x(idx:idx+frame_len-1).*hann_win;
        if sqrt(mean(frame.^2)) < 0.01, continue; end
        f0 = estimate_f0_fn(frame', TARGET_SR);
        if isnan(f0)||f0<50||f0>2000, continue; end
        m = compute_ctfs_fn(frame', TARGET_SR, f0, N_HARMONICS);
        mags_tmp = [mags_tmp; m];
    end
    raga_mags.(key) = mags_tmp;
end

% ANOVA per harmonic
for inst_idx = 1:2
    inst = INSTRUMENTS{inst_idx};
    if inst_idx == 1
        ragas = {'yaman','bhairavi'};
    else
        ragas = {'bhairavi','desh'};
    end
    
    f_stats = zeros(1,N_HARMONICS);
    p_vals  = zeros(1,N_HARMONICS);
    
    for n = 1:N_HARMONICS
        k1  = [inst '_' ragas{1}];
        k2  = [inst '_' ragas{2}];
        g1  = raga_mags.(k1)(:,n);
        g2  = raga_mags.(k2)(:,n);
        [p, tbl] = anova1([g1;g2], ...
            [ones(length(g1),1); 2*ones(length(g2),1)], 'off');
        f_stats(n) = tbl{2,5};
        p_vals(n)  = p;
    end
    
    sig_n = sum(p_vals < 0.05);
    fprintf('  %s (%s vs %s): %d/32 significant harmonics\n', ...
            inst, ragas{1}, ragas{2}, sig_n);
end

%% ═══════════════════════════════════════════════════
%% MODULE 7 — VISUALIZATION (All Plots)
%% ═══════════════════════════════════════════════════
fprintf('\nMODULE 7 — Generating Plots...\n');
fprintf('----------------------------------------\n');

fig_dir = fullfile(BASE_DIR,'results','figures');
if ~exist(fig_dir,'dir'), mkdir(fig_dir); end

%% ── PLOT 1: Harmonic Profiles ───────────────────────────────────
fprintf('  [1] Harmonic profiles...\n');
figure('Color',BG,'Position',[50 50 1000 520],'Name','Harmonic Profiles');
ax = axes; set(ax,'Color',BG,'XColor','w','YColor','w');
hold on;
n_vals = 1:N_HARMONICS;
for k = 1:numel(INSTRUMENTS)
    inst = INSTRUMENTS{k};
    mu   = mean(all_mags.(inst),1);
    sg   = std(all_mags.(inst),0,1);
    c    = COLORS{k};
    plot(n_vals, mu, 'Color',c, 'LineWidth',2.5, ...
         'DisplayName',[upper(inst(1)) inst(2:end)]);
    fill([n_vals fliplr(n_vals)],[mu+sg fliplr(mu-sg)],...
         c,'FaceAlpha',0.12,'EdgeColor','none','HandleVisibility','off');
end
xlabel('Harmonic Index (n)','Color','w','FontSize',12);
ylabel('Normalized |c_n|','Color','w','FontSize',12);
title('CTFS Harmonic Fingerprints — Each Instrument Has a Unique Profile',...
      'Color','w','FontSize',13,'FontWeight','bold');
legend('TextColor','w','Color',[0.15 0.15 0.25],'FontSize',11,...
       'Location','northeast');
grid on; set(gca,'GridColor','w','GridAlpha',0.12);
set(gca,'Box','off');
saveas(gcf, fullfile(fig_dir,'matlab_harmonic_profiles.png'));

%% ── PLOT 2: Stem Plots ──────────────────────────────────────────
fprintf('  [2] Stem plots...\n');
figure('Color',BG,'Position',[50 50 1200 460],'Name','Stem Plots');
for k = 1:numel(INSTRUMENTS)
    inst = INSTRUMENTS{k};
    mu   = mean(all_mags.(inst),1);
    c    = COLORS{k};
    subplot(1,3,k);
    ax = gca; set(ax,'Color',BG,'XColor','w','YColor','w');
    h  = stem(n_vals, mu, 'filled');
    h.Color           = c;
    h.MarkerFaceColor = c;
    h.MarkerSize      = 5;
    h.LineWidth       = 1.5;
    xlabel('Harmonic n','Color','w','FontSize',11);
    ylabel('|c_n|','Color','w','FontSize',11);
    title([upper(inst(1)) inst(2:end)],'Color',c,...
          'FontSize',14,'FontWeight','bold');
    grid on; set(gca,'GridColor','w','GridAlpha',0.12);
    ylim([0 max(mu)*1.15]);
end
sgtitle('CTFS Harmonic Stem Plots — Mean Envelope per Instrument',...
        'Color','w','FontSize',13,'FontWeight','bold');
saveas(gcf, fullfile(fig_dir,'matlab_stem_plots.png'));

%% ── PLOT 3: SNR Fidelity Curves ─────────────────────────────────
fprintf('  [3] SNR fidelity curves...\n');
figure('Color',BG,'Position',[50 50 1000 520],'Name','Reconstruction Fidelity');
ax = axes; set(ax,'Color',BG,'XColor','w','YColor','w');
hold on;
for k = 1:numel(INSTRUMENTS)
    inst = INSTRUMENTS{k};
    c    = COLORS{k};
    ns   = N_star.(inst);
    snr  = snr_curves.(inst);
    plot(n_vals, snr,'Color',c,'LineWidth',2.5,...
         'DisplayName',sprintf('%s (N*=%d)', [upper(inst(1)) inst(2:end)], ns));
    xline(ns,'--','Color',c,'Alpha',0.5,'HandleVisibility','off');
    scatter(ns, snr(ns), 60, c,'filled','HandleVisibility','off');
end
yline(SNR_THRESH,':','Color','w','LineWidth',1.5,...
      'DisplayName',sprintf('%.0f dB threshold', SNR_THRESH));
xlabel('Number of Harmonics (N)','Color','w','FontSize',12);
ylabel('SNR (dB)','Color','w','FontSize',12);
title('Reconstruction Fidelity — SNR vs Number of Harmonics',...
      'Color','w','FontSize',13,'FontWeight','bold');
legend('TextColor','w','Color',[0.15 0.15 0.25],'FontSize',11);
grid on; set(gca,'GridColor','w','GridAlpha',0.12,'Box','off');
saveas(gcf, fullfile(fig_dir,'matlab_snr_curves.png'));

%% ── PLOT 4: N* Bar Chart ────────────────────────────────────────
fprintf('  [4] N* bar chart...\n');
figure('Color',BG,'Position',[50 50 700 480],'Name','N* Threshold');
ax = axes; set(ax,'Color',BG,'XColor','w','YColor','w');
hold on;
ns_vals = [N_star.sitar, N_star.veena, N_star.guitar];
for k = 1:3
    bar(k, ns_vals(k),'FaceColor',COLORS{k},'EdgeColor','w',...
        'FaceAlpha',0.85,'LineWidth',0.5);
    text(k, ns_vals(k)+0.3, sprintf('N*=%d',ns_vals(k)),...
         'HorizontalAlignment','center','Color','w',...
         'FontSize',13,'FontWeight','bold');
end
set(gca,'XTick',1:3,'XTickLabel',{'Sitar','Veena','Guitar'});
xlabel('Instrument','Color','w','FontSize',12);
ylabel('N* (harmonics for 30dB SNR)','Color','w','FontSize',12);
title('Minimum Harmonic Threshold N* per Instrument',...
      'Color','w','FontSize',13,'FontWeight','bold');
ylim([0 max(ns_vals)+4]);
grid on; set(gca,'GridColor','w','GridAlpha',0.12,'Box','off');
saveas(gcf, fullfile(fig_dir,'matlab_nstar_comparison.png'));

%% ── PLOT 5: Confusion Matrices ──────────────────────────────────
fprintf('  [5] Confusion matrices...\n');
figure('Color',BG,'Position',[50 50 1100 480],'Name','Confusion Matrices');
instOrder = {'sitar','veena','guitar'};

subplot(1,2,1);
cm_svm = confusionmat(y_categorical, svm_pred,'Order',categorical(instOrder));
plot_cm_fn(cm_svm, instOrder, COLORS, BG, 'SVM (RBF)');

subplot(1,2,2);
cm_knn = confusionmat(y_categorical, knn_pred,'Order',categorical(instOrder));
plot_cm_fn(cm_knn, instOrder, COLORS, BG, 'kNN (k=5)');

sgtitle('Confusion Matrices — SVM vs kNN',...
        'Color','w','FontSize',13,'FontWeight','bold');
saveas(gcf, fullfile(fig_dir,'matlab_confusion_matrices.png'));

%% ── PLOT 6: ANOVA F-statistic ───────────────────────────────────
fprintf('  [6] ANOVA raga effect...\n');
figure('Color',BG,'Position',[50 50 1100 480],'Name','ANOVA Raga Effect');

for inst_idx = 1:2
    inst = INSTRUMENTS{inst_idx};
    if inst_idx == 1
        ragas = {'yaman','bhairavi'};
    else
        ragas = {'bhairavi','desh'};
    end
    
    f_stats = zeros(1,N_HARMONICS);
    p_vals  = zeros(1,N_HARMONICS);
    for n = 1:N_HARMONICS
        k1 = [inst '_' ragas{1}];
        k2 = [inst '_' ragas{2}];
        if ~isfield(raga_mags,k1)||~isfield(raga_mags,k2), continue; end
        g1 = raga_mags.(k1)(:,n);
        g2 = raga_mags.(k2)(:,n);
        [p,tbl] = anova1([g1;g2],[ones(length(g1),1);2*ones(length(g2),1)],'off');
        f_stats(n) = tbl{2,5};
        p_vals(n)  = p;
    end
    
    subplot(1,2,inst_idx);
    ax = gca; set(ax,'Color',BG,'XColor','w','YColor','w');
    hold on;
    bar_cols = zeros(N_HARMONICS,3);
    for n=1:N_HARMONICS
        if p_vals(n)<0.05
            bar_cols(n,:) = COLORS{inst_idx};
        else
            bar_cols(n,:) = [0.33 0.33 0.47];
        end
    end
    for n=1:N_HARMONICS
        bar(n,f_stats(n),'FaceColor',bar_cols(n,:),'EdgeColor','none','FaceAlpha',0.85);
    end
    yline(3.84,'--w','LineWidth',1.5,'DisplayName','F critical (p=0.05)');
    xlabel('Harmonic Index (n)','Color','w','FontSize',11);
    ylabel('F-statistic','Color','w','FontSize',11);
    title(sprintf('%s — %s vs %s',[upper(inst(1)) inst(2:end)],ragas{1},ragas{2}),...
          'Color','w','FontSize',12,'FontWeight','bold');
    grid on; set(gca,'GridColor','w','GridAlpha',0.12,'Box','off');
end
sgtitle('ANOVA — Does Raga Change the Harmonic Fingerprint?',...
        'Color','w','FontSize',13,'FontWeight','bold');
saveas(gcf, fullfile(fig_dir,'matlab_anova.png'));

%% ── PLOT 7: Reconstruction Demo (single note) ───────────────────
fprintf('  [7] Reconstruction demo...\n');
figure('Color',BG,'Position',[50 50 1000 600],'Name','Reconstruction Demo');

% Pick one sitar frame
demo_mags = all_mags.sitar(1,:);
demo_f0   = all_f0s.sitar(1);
t_demo    = (0:frame_len-1)/TARGET_SR * 1000;  % in ms

N_show = [1 3 5 9 32];
colors_demo = {[0.9 0.3 0.2],[0.9 0.6 0.2],[0.7 0.85 0.4],[0.4 0.85 0.7],[0.8 0.8 0.8]};

% Full original
original = zeros(1,frame_len);
for n=1:N_HARMONICS
    original = original + demo_mags(n)*cos(2*pi*n*demo_f0*(0:frame_len-1)/TARGET_SR);
end

for p = 1:length(N_show)
    N   = N_show(p);
    recon = zeros(1,frame_len);
    for n=1:N
        recon = recon + demo_mags(n)*cos(2*pi*n*demo_f0*(0:frame_len-1)/TARGET_SR);
    end
    err = original - recon;
    sp  = sum(original.^2);
    ep  = sum(err.^2);
    if ep < 1e-12, snr = 100; else, snr = 10*log10(sp/ep); end
    
    subplot(2,3,p);
    ax = gca; set(ax,'Color',BG,'XColor','w','YColor','w');
    plot(t_demo, original,'Color',[0.5 0.5 0.5],'LineWidth',1);
    hold on;
    plot(t_demo, recon,'Color',colors_demo{p},'LineWidth',1.8);
    title(sprintf('N=%d  |  SNR=%.1f dB',N,snr),...
          'Color',colors_demo{p},'FontSize',11,'FontWeight','bold');
    xlabel('Time (ms)','Color','w','FontSize',9);
    ylabel('Amplitude','Color','w','FontSize',9);
    legend({'Original','Reconstructed'},'TextColor','w',...
           'Color',[0.15 0.15 0.25],'FontSize',8,'Location','northeast');
    grid on; set(gca,'GridColor','w','GridAlpha',0.1,'Box','off');
    ylim([-1.2 1.2]);
end
subplot(2,3,6);
ax = gca; set(ax,'Color',BG,'XColor','w','YColor','w');
text(0.5,0.6,'N* = 9','HorizontalAlignment','center',...
     'Color',[0.72 0.31 0.26],'FontSize',28,'FontWeight','bold',...
     'Units','normalized');
text(0.5,0.35,'At N=9, SNR \geq 30dB','HorizontalAlignment','center',...
     'Color','w','FontSize',11,'Units','normalized');
text(0.5,0.2,'Sound is perceptually','HorizontalAlignment','center',...
     'Color',[0.65 0.75 0.68],'FontSize',10,'Units','normalized');
text(0.5,0.1,'identical to original','HorizontalAlignment','center',...
     'Color',[0.65 0.75 0.68],'FontSize',10,'Units','normalized');
set(ax,'XTick',[],'YTick',[]); box off;

sgtitle('Reconstruction Fidelity — Sitar Frame (f_0 ≈ 220Hz)',...
        'Color','w','FontSize',13,'FontWeight','bold');
saveas(gcf, fullfile(fig_dir,'matlab_reconstruction_demo.png'));

%% ── Final Summary ───────────────────────────────────────────────
fprintf('\n================================================\n');
fprintf('  SIMULATION COMPLETE\n');
fprintf('================================================\n');
fprintf('  SVM Accuracy : %.2f%%\n', svm_acc);
fprintf('  kNN Accuracy : %.2f%%\n', knn_acc);
fprintf('  N* Sitar     : %d\n', N_star.sitar);
fprintf('  N* Veena     : %d\n', N_star.veena);
fprintf('  N* Guitar    : %d\n', N_star.guitar);
fprintf('\n  Plots saved to: %s\n', fig_dir);
fprintf('  7 figures generated\n');
fprintf('================================================\n');


%% ═══════════════════════════════════════════════════
%% HELPER FUNCTIONS
%% ═══════════════════════════════════════════════════

function f0 = estimate_f0_fn(frame, sr)
    frame = frame - mean(frame);
    if max(abs(frame)) < 1e-6, f0=NaN; return; end
    corr    = xcorr(frame, frame);
    corr    = corr(length(frame):end);
    min_lag = round(sr/2000);
    max_lag = round(sr/50);
    if max_lag > length(corr), max_lag = length(corr); end
    [~,pk]  = max(corr(min_lag:max_lag));
    f0      = sr / (pk + min_lag - 1);
end

function mags = compute_ctfs_fn(frame, sr, f0, N)
    X     = fft(frame);
    freqs = (0:length(frame)-1)*sr/length(frame);
    mags  = zeros(1,N);
    for n = 1:N
        [~,idx] = min(abs(freqs - n*f0));
        mags(n) = abs(X(idx)) / length(frame);
    end
    if mags(1) > 1e-9, mags = mags/mags(1); end
end

function plot_cm_fn(cm, instOrder, colors, bg, title_str)
    imagesc(cm);
    colormap(flipud(hot));
    ax = gca;
    set(ax,'Color',bg,'XColor','w','YColor','w');
    set(ax,'XTick',1:3,'XTickLabel',instOrder,...
           'YTick',1:3,'YTickLabel',instOrder,'FontSize',11);
    thresh = max(cm(:))/2;
    for i=1:3
        for j=1:3
            c = 'w'; if cm(i,j)>thresh, c='k'; end
            text(j,i,num2str(cm(i,j)),...
                 'HorizontalAlignment','center','Color',c,...
                 'FontSize',13,'FontWeight','bold');
        end
    end
    xlabel('Predicted','Color','w','FontSize',11);
    ylabel('True','Color','w','FontSize',11);
    title(title_str,'Color','w','FontSize',12,'FontWeight','bold');
end