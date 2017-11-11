function CMC = evalCMC( rankingTable, labelGal, labelPrb )

[nGal, nPrb] = size(rankingTable);
CMC = zeros(min(nGal,20), 1);
for i = 1:nPrb
    RT = rankingTable(:, i);
    for j = 1:nGal
        idxGal = RT(j);
        if labelGal(idxGal) == labelPrb(i)
            CMC(j:end) = CMC(j:end) + 1;
            break
        end
    end
end
CMC = CMC/nPrb;

end

