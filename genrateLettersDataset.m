function [X]=genrateLettersDataset(state) 
% This function generates the 2D-Letters dataset given a state string
% "speed" , "position"
% Copyright (c) 2019 Idiap Research Institute, http://idiap.ch/
% Written by Hedi Fendri
% Supervised by Sylvain Calinon, http://calinon.ch/
% Created : 23/09/2019 
% Last modified: 30/09/2019
X=[];
numbLetters=26;
filename="dataset_letters.txt";
fid = fopen(filename, 'w');
for i=1:numbLetters
    demos=[];
    letter=char('A'+i-1);
    file=strcat('data/2Dletters/',letter,'.mat');
    load(file);
    nbSamples = length(demos);
    for n=1:nbSamples
        fprintf(fid,strcat(letter,'\n'));
        columns=[];
        if state=="position"
            columns=[columns demos{n}.pos(1,:), demos{n}.pos(2,:)];
        elseif state=="speed"
            columns=[columns diff(demos{n}.pos(1,:)), diff(demos{n}.pos(2,:))];
        else
            fprintf("INVALIDE STATE \n")
        end
        X = [X ;columns]; 
    end
end
X=X';
end