function build_features()
% BUILD_FEATURES  Module 4 — Feature Vector Construction (MATLAB)
%
% Loads CTFS .mat files from Module 2,
% computes mean harmonic envelopes per instrument,
% builds deviation vectors, saves feature matrix as CSV.
%
% Usage:
%   build_features()

    N_HARMONICS  = 32;
    BASE_DIR     = 'E:/CTFS_Harmonic_Fingerprinting';
    INSTRUMENTS  = {'sitar', 'veena', 'guitar'};

    feat_dir = fullfile(BASE_DIR, 'data', 'features');
    if ~exist(feat_dir, 'dir'), mkdir(feat_dir); end

    fprintf('MODULE 4 — FEATURE VECTOR CONSTRUCTION\n');
    fprintf('%s\n', repmat('=',1,45));

    % ── Step 1: Load all CTFS data ───────────────────────────────
    fprintf('\nLoading CTFS data...\n');
    raw_mags  = struct();
    raw_ragas = struct();

    for k = 1:numel(INSTRUMENTS)
        inst     = INSTRUMENTS{k};
        mat_path = fullfile(BASE_DIR, 'data', 'segments', inst, ...
                            [inst '_ctfs.mat']);

        if ~exist(mat_path, 'file')
            % Try ctfs folder (JSON was saved there by Python Module 2)
            fprintf('  [%s] .mat not found, skipping\n', inst);
            continue;
        end

        D = load(mat_path);
        raw_mags.(inst)  = D.all_mags;    % (frames x N_HARMONICS)
        raw_ragas.(inst) = D.raga;
        fprintf('  %-8s: %d frames loaded\n', inst, size(D.all_mags,1));
    end

    % ── Step 2: Compute mean envelopes ───────────────────────────
    fprintf('\nComputing mean harmonic envelopes...\n');
    mean_envs = struct();
    for k = 1:numel(INSTRUMENTS)
        inst = INSTRUMENTS{k};
        if ~isfield(raw_mags, inst), continue; end
        mean_envs.(inst) = mean(raw_mags.(inst), 1);  % (1 x N_HARMONICS)
        fprintf('  %-8s: mean envelope computed\n', inst);
    end

    % ── Step 3: Build deviation vectors ──────────────────────────
    fprintf('\nBuilding deviation vectors...\n');
    all_features = [];
    all_labels   = {};
    all_ragas    = {};

    for k = 1:numel(INSTRUMENTS)
        inst = INSTRUMENTS{k};
        if ~isfield(raw_mags, inst), continue; end

        mags     = raw_mags.(inst);
        mean_env = mean_envs.(inst);
        dev_vecs = mags - repmat(mean_env, size(mags,1), 1);

        n = size(dev_vecs, 1);
        all_features = [all_features; dev_vecs];

        for i = 1:n
            all_labels{end+1}  = inst;
            all_ragas{end+1}   = raw_ragas.(inst);
        end

        fprintf('  %-8s: %d deviation vectors\n', inst, n);
    end

    % ── Step 4: Save as CSV ───────────────────────────────────────
    feat_path = fullfile(feat_dir, 'feature_matrix.csv');
    fid = fopen(feat_path, 'w');

    % Header
    for n = 1:N_HARMONICS
        fprintf(fid, 'h%d,', n);
    end
    fprintf(fid, 'instrument,raga\n');

    % Rows
    for i = 1:size(all_features, 1)
        fprintf(fid, '%.6f,', all_features(i,:));
        fprintf(fid, '%s,%s\n', all_labels{i}, all_ragas{i});
    end
    fclose(fid);

    fprintf('\n  Feature matrix saved: %s\n', feat_path);
    fprintf('  Total samples : %d\n', size(all_features,1));
    fprintf('  Feature dims  : %d\n', N_HARMONICS);

    % ── Class distribution ────────────────────────────────────────
    fprintf('\n  Class Distribution:\n');
    fprintf('  %s\n', repmat('=',1,30));
    for k = 1:numel(INSTRUMENTS)
        inst  = INSTRUMENTS{k};
        count = sum(strcmp(all_labels, inst));
        pct   = 100 * count / numel(all_labels);
        fprintf('  %-8s : %4d (%.1f%%)\n', inst, count, pct);
    end
    fprintf('  %s\n', repmat('=',1,30));

    fprintf('\nMODULE 4 COMPLETE\n');
    fprintf('Proceed to Module 5 — Classification\n');
end