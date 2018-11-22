function I = constructI( d, nViews )
% I = [(N-1)E -E -E ... -E   \
%      -E (N-1)E -E ... -E   |
%               .            |
%               .            |-> N
%               .            |
%      -E -E ...    (N-1)E]; /

E = eye(d);
minusBigE = repmat(-E, nViews, nViews);
I = minusBigE + nViews*eye(nViews*d);

end

