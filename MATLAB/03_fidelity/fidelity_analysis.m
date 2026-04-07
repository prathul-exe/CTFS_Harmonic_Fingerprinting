function results = fidelity_analysis()
% FIDELITY_ANALYSIS  Module 3 — Reconstruction Fidelity (MATLAB)
%
% Loads CTFS data, reconstructs signals using N=1..32 harmonics,
% computes SNR(N) curve, finds N* at 30dB threshold.
% Plots and saves results.

    N_HARMONICS   = 32;
    SNR_THRESHOLD = 30.0;
    FRAME_MS      = 100;
    TARGET_SR     = 22050;
    BASE_DIR      = 'E:/CTFS_Harmonic_Fingerprinting';
    INSTRUMENTS   = {'sitar', 'veena', 'guitar'};

    results = struct();
    colors  = {'#B85042', '#A7BEAE', '#E7E8D1'};

    figure('Color', [0.1 0.1 0.18], 'Position', [100 100 900 550]);
    hold on;

    for k = 1:numel(INSTRUMENTS)
        inst = INSTRUMENTS{k};
        fprintf('\n[%s]\n', upper(inst));

        % Load CTFS .mat file
        mat_path = fullfile(BASE_DIR, 'data', 'segments', inst, ...
                            [inst '_ctfs.mat']);
        if ~exist(mat_path, 'file')
            fprintf('  NOT FOUND: %s\n', mat_path); continue;
        end
        D = load(mat_path);
        magnitudes = D.all_mags;   % (frames x N_HARMONICS)
        n_frames   = size(magnitudes, 1);
        fprintf('  %d frames loaded\n', n_frames);

        % Compute SNR curve averaged over frames
        snr_matrix = zeros(n_frames, N_HARMONICS);
        frame_len  = round(FRAME_MS * 1e-3 * TARGET_SR);
        t          = (0:frame_len-1) / TARGET_SR;

        for i = 1:n_frames
            mags = magnitudes(i,:);
            f0   = D.all_f0s(i);
            if f0 < 50 || f0 > 2000, continue; end

            % Original signal (all 32 harmonics)
            original = zeros(1, frame_len);
            for n = 1:N_HARMONICS
                original = original + mags(n) * cos(2*pi*n*f0*t);
            end

            % SNR for N=1..32
            for N = 1:N_HARMONICS
                recon = zeros(1, frame_len);
                for n = 1:N
                    recon = recon + mags(n) * cos(2*pi*n*f0*t);
                end
                err = original - recon;
                sig_pow = sum(original.^2);
                err_pow = sum(err.^2);
                if err_pow < 1e-12
                    snr_matrix(i,N) = 100;
                else
                    snr_matrix(i,N) = 10*log10(sig_pow / err_pow);
                end
            end
        end

        mean_snr = mean(snr_matrix, 1);
        std_snr  = std(snr_matrix, 0, 1);

        % Find N*
        N_star = N_HARMONICS;
        for N = 1:N_HARMONICS
            if mean_snr(N) >= SNR_THRESHOLD
                N_star = N; break;
            end
        end

        fprintf('  N* = %d | SNR at N=32: %.1f dB\n', N_star, mean_snr(end));

        results.(inst).mean_snr = mean_snr;
        results.(inst).N_star   = N_star;

        % Plot
        n_vals = 1:N_HARMONICS;
        c      = sscanf(colors{k}(2:end), '%2x%2x%2x', [1 3]) / 255;
        plot(n_vals, mean_snr, 'Color', c, 'LineWidth', 2.5, ...
             'DisplayName', sprintf('%s (N*=%d)', inst, N_star));
        fill([n_vals fliplr(n_vals)], ...
             [mean_snr+std_snr fliplr(mean_snr-std_snr)], ...
             c, 'FaceAlpha', 0.1, 'EdgeColor', 'none', ...
             'HandleVisibility', 'off');
        xline(N_star, '--', 'Color', c, 'Alpha', 0.5, ...
              'HandleVisibility', 'off');
    end

    % Threshold line
    yline(SNR_THRESHOLD, ':', 'Color', 'white', 'LineWidth', 1.5, ...
          'DisplayName', sprintf('%g dB threshold', SNR_THRESHOLD));

    % Formatting
    set(gca, 'Color', [0.1 0.1 0.18], 'XColor', 'white', ...
             'YColor', 'white', 'GridColor', 'white', ...
             'GridAlpha', 0.15);
    xlabel('Number of Harmonics (N)', 'Color', 'white', 'FontSize', 12);
    ylabel('SNR (dB)', 'Color', 'white', 'FontSize', 12);
    title('Reconstruction Fidelity — SNR vs N Harmonics', ...
          'Color', 'white', 'FontSize', 14, 'FontWeight', 'bold');
    legend('TextColor', 'white', 'Color', [0.15 0.15 0.25], ...
           'FontSize', 11);
    grid on; box off;

    % Save figure
    fig_dir = fullfile(BASE_DIR, 'results', 'figures');
    if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end
    saveas(gcf, fullfile(fig_dir, 'snr_curves.png'));
    fprintf('\n  Plot saved: results/figures/snr_curves.png\n');

    % Save N* table
    fid = fopen(fullfile(BASE_DIR, 'results', 'N_star_table.csv'), 'w');
    fprintf(fid, 'instrument,N_star\n');
    for k = 1:numel(INSTRUMENTS)
        inst = INSTRUMENTS{k};
        if isfield(results, inst)
            fprintf(fid, '%s,%d\n', inst, results.(inst).N_star);
        end
    end
    fclose(fid);

    fprintf('\nMODULE 3 COMPLETE\n');
    fprintf('Proceed to Module 4 — Feature Vectors\n');
end