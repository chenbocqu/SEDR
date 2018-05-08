function [ out ] = plot_accurcy( x,reco_rates,unit,l )

% 从第二步开始画迭代曲线
plot     ( x,reco_rates,'bo-',...
            'MarkerEdgeColor','b',...
            'MarkerFaceColor','w',...
            'MarkerSize',4);
        
xlabel      ( unit );
ylabel      ( '精度（%）' );
legend      ( l );

grid on;
hold on;

end

