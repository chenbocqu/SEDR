function [ out ] = plot_accurcy( x,reco_rates,unit,l )

% �ӵڶ�����ʼ����������
plot     ( x,reco_rates,'bo-',...
            'MarkerEdgeColor','b',...
            'MarkerFaceColor','w',...
            'MarkerSize',4);
        
xlabel      ( unit );
ylabel      ( '���ȣ�%��' );
legend      ( l );

grid on;
hold on;

end

