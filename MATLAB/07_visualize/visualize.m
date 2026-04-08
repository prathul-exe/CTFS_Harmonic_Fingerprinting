function visualize()
% VISUALIZE  Module 7 — Final Visualization (MATLAB)
%
% Generates all publication-quality plots:
%   1. Harmonic profile comparison
%   2. Deviation heatmaps
%   3. N* comparison bar chart
%   4. Stem plots per instrument
%
% Usage:
%   visualize()

    BASE_DIR    = 'E:/CTFS_Harmonic_Fingerprinting';
    INSTRUMENTS = {'sitar', 'veena', 'guitar'};
    N_HARMONICS = 32;
    BG          = [0.1 0.1 0.18];
    COLORS      = {[0.72 0.31 0.26], [0.65 0.75 0.68], [0.91 0.91 0.82]};

    fig_dir = fullfile(BASE_DIR, 'results', 'figures');
    if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

    fprintf('MODULE 7 — FINAL VISUALIZATION\n');
    fprintf('%s\n', repmat('=',1,45));

    % ── Plot 1: Harmonic Profile Comparison ──────────────────────
    fprintf('  [1] Harmonic profile comparison...\n');
    figure('Color', BG, 'Position', [50 50 1000 550]);
    ax = gca; ax.Color = BG; ax.XColor = 'w'; ax.YColor = 'w';
    hold on;
    n_vals = 1:N_HARMONICS;

    for k = 1:numel(INSTRUMENTS)
        inst = INSTRUMENTS{k};
        mat  = fullfile(BASE_DIR,'data','segments',inst,[inst '_ctfs.mat']);
        if ~exist(mat,'file'), continue; end
        D    = load(mat);
        mu   = mean(D.all_mags, 1);
        sg   = std(D.all_mags, 0, 1);
        c    = COLORS{k};
        plot(n_vals, mu, 'Color', c, 'LineWidth', 2.5, ...
             'DisplayName', [upper(inst(1)) inst(2:end)]);
        fill([n_vals fliplr(n_vals)], [mu+sg fliplr(mu-sg)], ...
             c, 'FaceAlpha', 0.12, 'EdgeColor','none', ...
             'HandleVisibility','off');
    end

    xlabel('Harmonic Index (n)',  'Color','w','FontSize',12);
    ylabel('Normalized |cₙ|',    'Color','w','FontSize',12);
    title('Harmonic Fingerprint Comparison', ...
          'Color','w','FontSize',13,'FontWeight','bold');
    legend('TextColor','w','Color',[0.15 0.15 0.25],'FontSize',11);
    grid on;
    set(gca,'GridColor','w','GridAlpha',0.12);
    saveas(gcf, fullfile(fig_dir,'harmonic_profiles.png'));
    close(gcf);
    fprintf('     Saved: harmonic_profiles.png\n');

    % ── Plot 2: Stem Plots ────────────────────────────────────────
    fprintf('  [2] Stem plots...\n');
    figure('Color', BG, 'Position', [50 50 1300 480]);

    for k = 1:numel(INSTRUMENTS)
        inst = INSTRUMENTS{k};
        mat  = fullfile(BASE_DIR,'data','segments',inst,[inst '_ctfs.mat']);
        if ~exist(mat,'file'), continue; end
        D    = load(mat);
        mu   = mean(D.all_mags, 1);
        c    = COLORS{k};

        subplot(1,3,k);
        ax = gca; ax.Color = BG; ax.XColor = 'w'; ax.YColor = 'w';
        h  = stem(n_vals, mu, 'filled');
        h.Color       = c;
        h.MarkerFaceColor = c;
        h.MarkerSize  = 5;
        h.LineWidth   = 1.5;
        xlabel('Harmonic n',      'Color','w','FontSize',11);
        ylabel('Normalized |cₙ|', 'Color','w','FontSize',11);
        title([upper(inst(1)) inst(2:end)], ...
              'Color', c, 'FontSize',13,'FontWeight','bold');
        grid on;
        set(gca,'GridColor','w','GridAlpha',0.12);
    end

    sgtitle('CTFS Harmonic Stem Plots — Mean Envelope per Instrument', ...
            'Color','w','FontSize',13,'FontWeight','bold');
    saveas(gcf, fullfile(fig_dir,'stem_plots.png'));
    close(gcf);
    fprintf('     Saved: stem_plots.png\n');

    % ── Plot 3: N* Bar Chart ──────────────────────────────────────
    fprintf('  [3] N* comparison bar chart...\n');
    csv_path = fullfile(BASE_DIR,'results','N_star_table.csv');
    T        = readtable(csv_path);

    figure('Color', BG, 'Position', [50 50 700 480]);
    ax = gca; ax.Color = BG; ax.XColor = 'w'; ax.YColor = 'w';
    hold on;

    for k = 1:height(T)
        bar(k, T.N_star(k), 'FaceColor', COLORS{k}, ...
            'EdgeColor','w','FaceAlpha',0.85,'LineWidth',0.5);
        text(k, T.N_star(k)+0.3, sprintf('N*=%d', T.N_star(k)), ...
             'HorizontalAlignment','center','Color','w', ...
             'FontSize',12,'FontWeight','bold');
    end

    ax.XTick      = 1:height(T);
    ax.XTickLabel = T.instrument;
    xlabel('Instrument',                   'Color','w','FontSize',12);
    ylabel('N* (harmonics for 30dB SNR)',  'Color','w','FontSize',12);
    title('Minimum Harmonic Threshold N* per Instrument', ...
          'Color','w','FontSize',13,'FontWeight','bold');
    ylim([0 max(T.N_star)+4]);
    grid on; grid minor;
    set(gca,'GridColor','w','GridAlpha',0.12);
    saveas(gcf, fullfile(fig_dir,'N_star_comparison.png'));
    close(gcf);
    fprintf('     Saved: N_star_comparison.png\n');

    % ── Plot 4: Deviation Heatmaps ────────────────────────────────
    fprintf('  [4] Deviation heatmaps...\n');
    figure('Color', BG, 'Position', [50 50 1300 480]);

    for k = 1:numel(INSTRUMENTS)
        inst = INSTRUMENTS{k};
        mat  = fullfile(BASE_DIR,'data','segments',inst,[inst '_ctfs.mat']);
        if ~exist(mat,'file'), continue; end
        D    = load(mat);
        mags = D.all_mags;
        mu   = mean(mags,1);
        dev  = mags - mu;
        % Sample 200 frames
        n_show = min(200, size(dev,1));
        dev    = dev(1:n_show, :);

        subplot(1,3,k);
        imagesc(dev');
        colormap(redblue());
        clim([-0.5 0.5]);
        ax = gca; ax.Color = BG; ax.XColor = 'w'; ax.YColor = 'w';
        xlabel('Frame index','Color','w','FontSize',10);
        ylabel('Harmonic n', 'Color','w','FontSize',10);
        title([upper(inst(1)) inst(2:end)], ...
              'Color', COLORS{k}, 'FontSize',12,'FontWeight','bold');
    end

    sgtitle('CTFS Deviation Heatmaps', ...
            'Color','w','FontSize',13,'FontWeight','bold');
    saveas(gcf, fullfile(fig_dir,'deviation_heatmaps.png'));
    close(gcf);
    fprintf('     Saved: deviation_heatmaps.png\n');

    fprintf('\n%s\n', repmat('=',1,45));
    fprintf('ALL PLOTS GENERATED\n');
    fprintf('Check: %s\n', fig_dir);
    fprintf('%s\n', repmat('=',1,45));
end


% ── Helper: Red-Blue colormap ─────────────────────────────────────
function c = redblue()
    n = 64;
    r = [linspace(0.2,1,n/2), linspace(1,0.9,n/2)]';
    g = [linspace(0.3,0.9,n/2), linspace(0.9,0.3,n/2)]';
    b = [linspace(0.9,0.9,n/2), linspace(0.9,0.2,n/2)]';
    c = [r g b];
end