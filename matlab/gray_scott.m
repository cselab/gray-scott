clc; clear

F= 0.04;
kappa = 0.06;
% Diffusion constants for Gray-Scott kinetics
D_a = 1e-2;
D_s = 5e-3;

fq = 0.90;

% dimensions
Lx = 5.50;
Ly = 5.50;
Nx = 257;
Ny = 257;
dx = Lx/(Nx-1);
dy = Ly/(Ny-1);
Tmax = 1000;
Tplot = 10;

[x,y] = meshgrid(1:Nx,1:Ny);
x = dx*x;
y = dy*y;

[a,s] = initial_condition_2(Nx,Ny);

ah = zeros(Nx, Ny);
sh = zeros(Nx, Ny);

a = update_ghosts(a);
s = update_ghosts(s);

pcolor(x,y,a);
shading interp;
drawnow;

fa = D_a/dx^2; 
fs = D_s/dx^2;
dt = fq/(4.0*fa);
fprintf('dt = %e\n', dt)

for t = 2:Tmax
    for i = 2:Nx-1
        
        im = i-1; 
        ip = i+1;
        
        for j = 2:Ny-1
            
            jm = j-1; 
            jp = j+1;
            
            a_xx = a(ip,j) + a(i,jp) + a(im,j) + a(i,jm) - 4*a(i,j);
            s_xx = s(ip,j) + s(i,jp) + s(im,j) + s(i,jm) - 4*s(i,j);
            
            ah(i,j) = a(i,j) + dt*( fa*a_xx - a(i,j) * s(i,j)^2 + F*( 1-a(i,j) ) );
            sh(i,j) = s(i,j) + dt*( fs*s_xx + a(i,j) * s(i,j)^2 - ( F + kappa) * s(i,j) );
        end
    end
      
    a(2:end-1,2:end-1) = ah(2:end-1,2:end-1); 
    s(2:end-1,2:end-1) = sh(2:end-1,2:end-1);
    
    a = update_ghosts(a);
    s = update_ghosts(s);
    
    if(mod(t,Tplot)==0)
        pcolor(x,y,a)
        title(['t= ' num2str(t)])
        shading interp
        drawnow
    end
end


function v = update_ghosts(v)
    v(1,:) = v(end-1,:);
    v(:,1) = v(:,end-1);
    v(end,:) = v(2,:);
    v(:,end) = v(:,2);
end


function [a,s] = initial_condition_2(Nx,Ny)
    a = ones(Nx,Ny)/2 + 0.5*unifrnd(0,1,Nx,Ny);
    s = ones(Nx,Ny)/4 + 0.5*unifrnd(0,1,Nx,Ny);
end


function [a,s] = initial_condition_1(Nx,Ny)
    a = ones(Nx,Ny)  + 0.01*unifrnd(0,1,Nx,Ny);
    s = zeros(Nx,Ny) + 0.01*unifrnd(0,1,Nx,Ny);
    
    for i = 1:Nx
        for j = 1:Ny
            if( (i>Nx/3) && (i<2*Nx/3) && (j>Ny/3) && (j<2*Ny/3))
                a(i,j) = 0.5 + 0.5*rand; 
                s(i,j) = 0.25 + 0.25*rand;
            end
        end
    end
end