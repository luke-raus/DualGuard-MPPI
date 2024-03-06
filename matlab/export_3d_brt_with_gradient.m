function export_3d_brt_with_gradient(filename, g, value, init_value, pdDims, dynSys)

% NOTE: This function is hardcoded for having the 3rd dimension be periodic
% because I'm pushing to get MVP to work; generalization will happen later
assert(pdDims == 3)

% Hint towards code that generalizes: from helperOC/valFuncs/computeOptTraj
% Basic idea is to represent dimension slicing inputs as cells of strings
%   clns = repmat({':'}, 1, g.dim);
%   BRS_at_t = data(clns{:}, tEarliest);

%% Get value grid at final time step and ensure periodicity

% TODO: Explore using helperOC/valFuncs/augmentPeriodicData for all this.......

% Value grid is final one along time dimension
value = value(:,:,:,end);


%% Sample optimal control at each grid point

% This is heavily based on code from helperOC/valFuncs/computeOptTraj

grid_axes = g.vs;

% Get gradients of value function for passing in deriv to optCtrl function
Deriv = computeGradients(g, value);     % This does level-set magic, apparently

theta_deriv = Deriv{3};


%% Handle periodicity
% Append the index=1 values of dim 3 to end since phi is periodic
      value(:,:,end+1) =       value(:,:,1);
 init_value(:,:,end+1) =  init_value(:,:,1);
theta_deriv(:,:,end+1) = theta_deriv(:,:,1);

% Save grid_axes as a cell array. Upon loading with scipy's mat loader
% using simplify_cells=True, we get a 2d array, and calling tuple() on
% this gives us the structure (1d arrays within tuple) that the scipy
% interpolating function wants (or, presumably, easy casting to a tensor
% for the pytorch version.
grid_axes{3}(end+1) = grid_axes{3}(end) + g.dx(3);

%% Write variables to file

save(filename, 'value', 'init_value', 'theta_deriv', 'grid_axes')

end