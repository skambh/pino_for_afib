function [Vsav,Wsav, tend, geom]=test()
close all
clear all

num_iter = 50;
tend = 145;
ncells=100;
dt=0.005; % AU, time step for finite differences solver
Vsav = zeros(num_iter, ncells, ncells, tend);
Wsav = zeros(num_iter, ncells, ncells, tend);
X = ncells + 2; % to allow boundary conditions implementation
Y = ncells + 2;
iscyclic=0;
flagmovie=0;
h = 0.1; % mm cell length
geom = (1:X) * h;


for iter = 1:num_iter
    
    
    Dfac = 1;
    % a = 0.002 + rand * (0.01 - 0.002); % Random a in [0.01, 0.02]
    b = 0.075 + rand * (0.15 - 0.075); % Random b in [0.075, 0.15]
    % D0 = 0.02 + rand * (0.1 - 0.02); % Random D0 in [0.02, 0.1]
    a = 0.01;
    % b = 0.15;
    D0 = 0.1;
    k = 8.0;
    mu1 = 0.2;
    mu2 = 0.3;
    epsi = 0.002;
    X = ncells + 2; % to allow boundary conditions implementation
    fibloc=[floor(X/3) ceil(X/3+X/5)]; % location of (square) heterogeneity
    
    D = D0*ones(X,X);
    D(fibloc(1):fibloc(2),fibloc(1):fibloc(2))=D0*Dfac;
    
    
    stimgeo=false(X,Y);
    stimgeo(1:5,:)=true; % indices of cells where external stimulus is felt
    
    crossfgeo=false(X,Y); % extra stimulus to generate spiral wave
    crossfgeo(:,1:floor(X/3))=true;
    tCF=42; % time (AU) at which the extra stimulus is applied
    
    
    gathert=round(1/dt); % number of iterations at which V is outputted
    
    tstar=tCF; % time, AU, at which the data starts being saved (because the 
    stimdur=1; % AU, duration of stimulus
    Ia=0.12; % AU, value for Istim when cell is stimulated
    
    V(1:X,1:Y)=0; % initial V
    W(1:X,1:Y)=0.01; % initial W
    
    % Vsav=zeros(ncells,ncells,ceil((tend-tstar)/gathert)); % array where V will be saved during simulation
    % Wsav=zeros(ncells,ncells,ceil((tend-tstar)/gathert)); % array where W will be saved during simulation
    
    ind=0; %iterations counter
    
    y=zeros(2,size(V,1),size(V,2));
    
    % for loop for explicit RK4 finite differences simulation
    for t=dt:dt:tend % for every timestep
        ind=ind+1; % count interations
            % stimulate at every BCL time interval for ncyc times
            if t<=stimdur
                Istim=Ia*stimgeo; % stimulating current
            elseif t>=tCF&&t<=tCF+stimdur
                Istim=Ia*crossfgeo;
            else
                Istim=zeros(X,Y); % stimulating current
            end
            
            % 4-step explicit Runga-Kutta implementation
            y(1,:,:)=V;
            y(2,:,:)=W;
            k1=AlPan(y,Istim);
            k2=AlPan(y+dt/2.*k1,Istim);
            k3=AlPan(y+dt/2.*k2,Istim);
            k4=AlPan(y+dt.*k3,Istim);
            y=y+dt/6.*(k1+2*k2+2*k3+k4);
            V=squeeze(y(1,:,:));
            W=squeeze(y(2,:,:));
                          
            % rectangular boundary conditions: no flux of V
            if  ~iscyclic % 1D cable
                V(1,:)=V(2,:);
                V(end,:)=V(end-1,:);
                V(:,1)=V(:,2);
                V(:,end)=V(:,end-1);
            else % periodic boundary conditions in x, y or both
                % set up later - need to amend derivatives calculation too
            end
            
            % At every gathert iterations, save V value for plotting
            if t>=tstar&&mod(ind,gathert)==0
                % save values
                Vsav(iter, :,:,round(ind/gathert))=V(2:end-1,2:end-1)';
                Wsav(iter, :,:,round(ind/gathert))=W(2:end-1,2:end-1)';
                % Vsav(:,:,round(ind/gathert))=V(2:end-1,2:end-1)';
                % Wsav(:,:,round(ind/gathert))=W(2:end-1,2:end-1)';
                % show (thicker) cable
                if flagmovie
                    subplot(2,1,1)
                    imagesc(V(2:end-1,2:end-1)',[0 1])
                    hold all
                    if Dfac<1 % there is a heterogeneity
                        rectangle('Position',[fibloc(1) fibloc(1) ...
                            fibloc(2)-fibloc(1) fibloc(2)-fibloc(1)]);
                    end
                    axis image
                    set(gca,'FontSize',14)
                    xlabel('x (voxels)')
                    ylabel('y (voxels)')
                    set(gca,'FontSize',14)
                    title(['V (AU) - Time: ' num2str(t,'%.0f') ' ms'])
                    colorbar
                    hold off
                    
                    subplot(2,1,2)
                    imagesc(W(2:end-1,2:end-1)',[0 1])
                    hold all
                    if Dfac<1 % there is a heterogeneity
                        rectangle('Position',[fibloc(1) fibloc(1) ...
                            fibloc(2)-fibloc(1) fibloc(2)-fibloc(1)]);
                    end
                    axis image
                    set(gca,'FontSize',14)
                    xlabel('x (voxels)')
                    ylabel('y (voxels)')
                    set(gca,'FontSize',14)
                    title(['V (AU) - Time: ' num2str(t,'%.0f') ' ms'])
                    colorbar
                    set(gca,'FontSize',14)
                    title('W (AU)')
                    colorbar
                    pause(0.01)
                    hold off
                end
            end
    end
    % Vsav_all{iter} = Vsav;
    % Wsav_all{iter} = Wsav;
end
close all

function dydt = AlPan(y,Istim)
    % global a k mu1 mu2 epsi b h D
    
    V=squeeze(y(1,:,:));
    W=squeeze(y(2,:,:));
    
    [gx,gy]=gradient(V,h);
    [Dx,Dy]=gradient(D,h);
    
    dV=4*D.*del2(V,h)+Dx.*gx+Dy.*gy; % extra terms to account for heterogeneous D
    dWdt=(epsi + mu1.*W./(mu2+V)).*(-W-k.*V.*(V-b-1));
    dVdt=(-k.*V.*(V-a).*(V-1)-W.*V)+dV+Istim;
    dydt(1,:,:)=dVdt;
    dydt(2,:,:)=dWdt;
end
end
