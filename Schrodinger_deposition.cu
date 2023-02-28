function X = SimulateDeposition(sys_length, t_max, deposition_rate, rule)
format long;
L = sys_length;
rate = deposition_rate;
temp_max = 0; % dummy variable
MaxHeight = 100;
for i = 1:L
    h(i) = 0;
end
for i = 1:t_max
    width(i) = 0;
end
% generating an array of random numbers
A =  ceil(L*rand(t_max,rate));
% the simulation
if((rule == 1) | (rule == 2))
    if (rule == 1)
        rule_name = 'Simple';
    else
        rule_name = 'Ballistic';
    end
for t = 1: t_max
    for k = 1 : rate
            i = A(t,k);
        if (rule == 1)
            h(i) = h(i) + 1;
            hit(i,h(i)) = 1;
        else
            if (i == 1)
                h(i) = max(h(i)+1,h(i+1));
                hit(i,h(i)) = 1;
            elseif (i == L)
                h(i) = max(h(i-1),h(i)+1);
                hit(i,h(i)) = 1;
            else    
                temp_max = max(h(i-1),h(i+1));
                h(i) = max(temp_max,h(i)+1);
                hit(i,h(i)) = 1;
            end
        end    
    end
    
    width(t) = std(h,1); 
    
    % Printing h(i,t) on the screen to verify the results
    % (note: You can uncomment the following 3 lines if you want to check)
    % fprintf('Time : %d\n', t);
    % fprintf('Width Fluctuation : %d\n', width(t));
    % h
end
% Plotting the width as a function of time using semilogx function
figure('Name','Width Fluctuation as a Function of Time in 1 Dimensional Lattice');
semilogx(width(1:t_max),'s',...
                               'MarkerEdgeColor','r',...
                               'MarkerFaceColor','b',...
                               'MarkerSize',4)
xlabel('Log10(Time)','FontSize',12)
ylabel('W(L,t)','FontSize',12)
title(['L = ',num2str(L),'   Time Max = ',num2str(t_max), '   Rate = ', num2str(rate), '   Rule = ', rule_name],'FontSize',12)
grid on
print -djpeg WidthFluc_1D.jpg;
% Plotting the surface
figure
for iatom=1:L
   for jatom=h(iatom):-1:1
       if (hit(iatom,jatom)==1)
        plotatom(iatom,jatom,'b')
       end
       hold on
   end
end
axis([1 L 0 (max(h)+10)])
xlabel('column i','FontSize',12)
ylabel('h(i)','FontSize',12)
title(['Surface Plot of L = ', num2str(L)],'FontSize',12)
print -djpeg Surface_Plot1D.jpg;
else
fprintf('\nInvalid Rule\n');
fprintf('\nRule = 1 for Simple Rule where h(i,t+1) = h(i,t) + 1\n');
fprintf('\nRule = 2 for Ballistic Rule where h(i,t+1) = MAX[h(i-1,t),h(i,t)+1,h(i+1,t)]\n');
end
%-------------------------------------------------------------------------%
function plotatom(x0,y0,color)
% plot one atom in the specified color, centered at (x0,y0) with radius r
r = 0.5;
theta = 0:0.1:6.29;
x = x0 + r*cos(theta);
y = y0 + r*sin(theta);
fill(x,y,color)
plot(x,y,color)
