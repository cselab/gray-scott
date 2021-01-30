clc; clear


function v = update_ghosts(v)
    v(1,:) = v(end-1,:);
    v(:,1) = v(:,end-1);
    v(end,:) = v(2,:);
    v(:,end) = v(:,2);
end

F= 1e-4;
kappa = 1e-4;
% Diffusion constants for Gray-Scott kinetics
D_a = 2e-04; D_s = 1e-04;

% dimensions
Lx = 5.50;
Ly = 5.50;
Nx = 257;
Ny = 257;
dx = Lx/(Nx-1);
dy = Ly/(Ny-1);
Tmax = 200000;
Tplot=20;

%arrays
a = zeros(Nx, Ny);
s = zeros(Nx, Ny);
ah = zeros(Nx, Ny);
sh = zeros(Nx, Ny);
x = zeros(Nx, Ny);
y = zeros(Nx, Ny);

% initialization 
for i = 1:Nx
    for j = 1:Ny
        x(i,j) = i*dx; 
        y(i,j) = j*dy;
        a(i,j) = 1.0 + 0.01*rand;
        s(i,j) = 0.0 + 0.01*rand;
      
        if( (i>Nx/3) && (i<2*Nx/3) && (j>Ny/3) && (j<2*Ny/3))
            a(i,j) = 0.5 + 0.005*rand; 
            s(i,j) = 0.25 + 0.0025*rand;
        end
    end
end

a = update_ghosts(a);
s = update_ghosts(s);


s(1,:) = s(Nx-1,:);
s(:,1) = s(:,Ny-1);
s(Nx,:) = s(2,:);
s(:,Ny) = s(:,2);

pcolor(x,y,a);
shading interp;
drawnow;

 
% PDE parameters for Gray-Scott kinetics
fa = D_a/dx^2; 
fs = D_s/dx^2;
fq = 0.90;
dt = fq/(4.0*fa);
printf('dt = %e\n', dt)

for t = 2:Tmax
    for i = 2:Nx-1
        
        im = i-1; 
        ip = i+1;
        
        for j = 2:Ny-1
            
            jm = j-1; 
            jp = j+1;
            
            a(i,j) = a(i,j); 
            a(ip,j) = a(ip,j);
            a(im,j) = a(im,j);
            a(i,jp) = a(i,jp);
            a(i,jm) = a(i,jm);
            
            s(i,j) = s(i,j); 
            s(ip,j) = s(ip,j);
            s(im,j) = s(im,j);
            s(i,jp) = s(i,jp);
            s(i,jm) = s(i,jm);
            
            a_xx = a(ip,j) + a(i,jp) + a(im,j) + a(i,jm) - 4*a(i,j);
            s_xx = s(ip,j)+s(i,jp)+s(im,j)+s(i,jm)-4*s(i,j);
            
            ah(i,j) = a(i,j) + dt*(fa * (a_xx) - a(i,j) * s(i,j)^2 + F * (1-a(i,j)) );
            sh(i,j) = s(i,j) + dt*(fs * (s_xx) + a(i,j) * s(i,j)^2 - (F + kappa) * s(i,j) );
        end
    end
      
    a(2:end-1,2:end-1) = ah(2:end-1,2:end-1); 
    s(2:end-1,2:end-1) = sh(2:end-1,2:end-1);
    
    a = update_ghosts(a);
    s = update_ghosts(s);
    
    printf('iter = %e\n', t)
    if(mod(t,Tplot)==0)
        pcolor(x,y,a)
        shading interp
        drawnow
    end
end
