//  Código basado en RLTutorial (tutorial.cpp) de profesor Julio Godoy
#include <iostream>
#include <vector>
#include <fstream>
#include <stdlib.h>
#include <math.h>
#include <cmath>
#include <cstdlib>
#include <time.h>
#include <string.h>

using namespace std;

// Variables globales para el ambiente y el agente
int height_grid, width_grid, action_taken, action_taken2,current_episode;
int blocked[100][100];
float cum_reward,Qvalues[100][100][4], reward[100][100],finalrw[50000];
int init_x_pos, init_y_pos, goalx, goaly, x_pos,y_pos, prev_x_pos, prev_y_pos,i,j,k;
ofstream reward_output;

//////////////
//Setting value for learning parameters
int action_sel=2; // 1 es greedy, 2 es e-greedy

int environment= 2; // 1 es el grid chico, 2 es el precipicio (Cliff walking)
int algorithm = 2; //1 es Q-learning, 2 es Sarsa
int stochastic_actions=1; // 0 para acciones normales, 1 para estocasticas

int num_episodes=2000; //total de episodios de entrenamiento
float learn_rate=0.1; // Tasa de aprendizaje (alpha)
float disc_factor=0.99; // Factor de descuento (gamma)
float exp_rate=0.05; // Epsilon para e-greedy (PUNTO 2)
///////////////


void Initialize_environment()
{
    if(environment==1)
    {
        height_grid= 3;
        width_grid=4;
        goalx=3;
        goaly=2;
        init_x_pos=0;
        init_y_pos=0;
    }
    
    if(environment==2)
    {
        height_grid= 4;
        width_grid=12;
        goalx=11;
        goaly=0;
        init_x_pos=0;
        init_y_pos=0;
    }
    
    // Inicializar recompensas y Q-values
    for(i=0; i < width_grid; i++)
    {
        for(j=0; j< height_grid; j++)
        {
            if(environment==1)
            {
                reward[i][j]=-0.04;
                blocked[i][j]=0;
            }
            
            if(environment==2)
            {
                reward[i][j]=-1;
                blocked[i][j]=0;
            }
            
            for(k=0; k<4; k++)
            {
                Qvalues[i][j][k]=0; // Empezamos con todo en 0
            }
        }
    }
    
    // Definir estados especiales (meta, obstaculos, etc)
    if(environment==1)
    {
        reward[goalx][goaly]=1;
        reward[goalx][(goaly-1)]=-1;
        blocked[1][1]=1; // Pared
    }
    
    if(environment==2)
    {
        reward[goalx][goaly]=1;
        for(int h=1; h<goalx;h++)
        {   
            reward[h][0]=-100; // El precipicio
        }
    }
}

// PUNTO 2: Implementación de Epsilon-Greedy
int action_selection()
{ 
    if(action_sel==1) // Greedy simple
    {
        float max_q = -99999.0;
        int best_action = 0;
        for(int act=0; act<4; act++){
            if(Qvalues[x_pos][y_pos][act] > max_q){
                max_q = Qvalues[x_pos][y_pos][act];
                best_action = act;
            }
        }
        return best_action;
    }
    
    if(action_sel==2)// Epsilon-greedy
    {
        float random_val = (float)rand() / RAND_MAX; // num aleatorio entre 0.0 y 1.0
        
        if(random_val < exp_rate) // Explorar: elegir una accion al azar
        {
            return rand()%4;
        }
        else // Explotar: elegir la mejor accion conocida
        {
            float max_q = -99999.0;
            int best_action = 0;
            // No manejamos empates, solo tomamos el primero que encontremos
            for(int act=0; act<4; act++){
                if(Qvalues[x_pos][y_pos][act] > max_q){
                    max_q = Qvalues[x_pos][y_pos][act];
                    best_action = act;
                }
            }
            return best_action;
        }
    }
    return 0; // No deberia llegar aqui
}

void move(int action)
{
    prev_x_pos=x_pos; 
    prev_y_pos=y_pos;
    
    // PUNTO 3: Implementación de acciones estocásticas
    if(stochastic_actions == 1)
    {
        float random_val = (float)rand() / RAND_MAX;
        
        // 80% de las veces se mueve donde se le indica (no hacemos nada)
        // 10% se mueve a la derecha de la accion deseada
        if (random_val < 0.1) 
        {
            action = (action + 1) % 4; // Derecha
        }
        // 10% se mueve a la izquierda
        else if (random_val < 0.2)
        {
            action = (action + 3) % 4; // Izquierda
        }
    }
    
    // Mover el agente
    if(action==0) // Arriba
    {
        if((y_pos < height_grid-1) && (blocked[x_pos][y_pos+1]==0)) { y_pos=y_pos+1; }
    }
    else if(action==1)  // Derecha
    {
        if((x_pos < width_grid-1) && (blocked[x_pos+1][y_pos]==0)) { x_pos=x_pos+1; }
    }
    else if(action==2)  // Abajo
    {
        if((y_pos > 0) && (blocked[x_pos][y_pos-1]==0)) { y_pos=y_pos-1; }
    }
    else if(action==3)  // Izquierda
    {
        if((x_pos > 0) && (blocked[x_pos-1][y_pos]==0)) { x_pos=x_pos-1; }
    }
}

// PUNTO 4: Implementación de la actualización de Q-Learning
void update_q_prev_state() 
{
    // Encontrar el max Q del estado actual para la formula
    float max_q_current = -99999.0;
    for(int act=0; act<4; act++){
        if(Qvalues[x_pos][y_pos][act] > max_q_current){
            max_q_current = Qvalues[x_pos][y_pos][act];
        }
    }
    
    float target;
    // Chequeamos si el estado actual es terminal
    if(!( ((x_pos==goalx)&&(y_pos==goaly)) ||((environment==1)&&(x_pos==goalx)&&(y_pos==(goaly-1)))||((environment==2)&&(x_pos>0)&&(x_pos<goalx)&&(y_pos==0))) )
    {
        // Si no es terminal, usamos la formula completa
        target = reward[x_pos][y_pos] + disc_factor * max_q_current;
    }
    else // Si es terminal, el valor futuro es 0
    {
        target = reward[x_pos][y_pos];
    }
    
    // Actualizamos el Q-value del estado ANTERIOR
    float old_q_value = Qvalues[prev_x_pos][prev_y_pos][action_taken];
    float td_error = target - old_q_value; // Error de diferencia temporal
    Qvalues[prev_x_pos][prev_y_pos][action_taken] = old_q_value + learn_rate * td_error;
}

// PUNTO 4: Implementación de la actualización de SARSA
void update_q_prev_state_sarsa()
{
    // En SARSA, no usamos el max_q, sino el q de la siguiente accion que ya elegimos
    float q_next_action = Qvalues[x_pos][y_pos][action_taken2];
    
    float target;
    // Chequeamos si el estado actual es terminal
    if(!( ((x_pos==goalx)&&(y_pos==goaly)) ||((environment==1)&&(x_pos==goalx)&&(y_pos==(goaly-1)))||((environment==2)&&(x_pos>0)&&(x_pos<goalx)&&(y_pos==0))     ) )
    {
        // Si no es terminal, usamos la formula completa de SARSA
        target = reward[x_pos][y_pos] + disc_factor * q_next_action;
    }
    else // Si es terminal, el valor futuro es 0
    {
        target = reward[x_pos][y_pos];
    }
    
    // Actualizamos el Q-value del estado ANTERIOR
    float old_q_value = Qvalues[prev_x_pos][prev_y_pos][action_taken];
    float td_error = target - old_q_value;
    Qvalues[prev_x_pos][prev_y_pos][action_taken] = old_q_value + learn_rate * td_error;
}

void Qlearning()
{
   action_taken = action_selection();
   move(action_taken);
   cum_reward=cum_reward+reward[x_pos][y_pos];
   update_q_prev_state();
}

void Sarsa()
{
    // La accion (action_taken) ya fue elegida antes de entrar aqui
    move(action_taken);
    cum_reward=cum_reward+reward[x_pos][y_pos];
    action_taken2 = action_selection(); // Elegimos la SIGUIENTE accion
    update_q_prev_state_sarsa(); // Actualizamos Q usando la siguiente accion
    action_taken = action_taken2; // La siguiente accion se vuelve la actual para el proximo paso
}

int main(int argc, char* argv[])
{
    srand(time(NULL)); // Semilla para numeros aleatorios
    
    // PUNTO 1: Abrir archivo para guardar recompensas
    reward_output.open("Rewards.txt");
    if (!reward_output.is_open()) {
        cout << "Error: No se pudo abrir el archivo Rewards.txt" << endl;
        return 1; // Salir si hay error
    }

    Initialize_environment();

    for(i=0;i<num_episodes;i++)
    {
        x_pos=init_x_pos;
        y_pos=init_y_pos;
        cum_reward=0;
        
        if(algorithm==2) // Para SARSA, elegimos la primera accion antes de empezar el loop
        {
            action_taken = action_selection();
        }
        
        // Loop principal del episodio, termina cuando llega a un estado terminal
        while(!( ((x_pos==goalx)&&(y_pos==goaly)) ||((environment==1)&&(x_pos==goalx)&&(y_pos==(goaly-1)))||((environment==2)&&(x_pos>0)&&(x_pos<goalx)&&(y_pos==0))     ) )
        {
            if(algorithm==1)
            {
                Qlearning();
            }
            if(algorithm==2)
            {
                Sarsa();
            }
        }

        finalrw[i]=cum_reward;
        
        // Imprimir progreso en la consola
        cout << "Episodio: " << i << ", Recompensa Acumulada: " << finalrw[i] << endl;
        
        // Guardar en el archivo
        reward_output << i << "," << finalrw[i] << endl;
    }
    
    reward_output.close();
    cout << "\nEntrenamiento finalizado. Datos guardados en Rewards.txt" << endl;

    return 0;
}

 