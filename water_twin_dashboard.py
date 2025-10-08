"""
Digital Twin of a Water Distrobution system personal project
"""

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import numpy as np
import pandas as pd
from datetime import datetime
from collections import deque
import threading
import time
import wntr

#wntr setup
class WaterDistributionSimulator:
    
    def __init__(self):
        self.wn = None
        self.sim = None
        self.results = None
        self.time_step = 0
        self.reservoir_head = 100.0
        self.demands = {'J2': 0.008, 'J3': 0.006}  # m3/s 
        self.initialize_network()

    #function to create network  
    def initialize_network(self):
        self.wn = wntr.network.WaterNetworkModel()
        self.wn.add_reservoir(name='R1', base_head=self.reservoir_head)
        
        self.wn.add_junction(name='J1', base_demand=0.005, elevation=50.0)
        self.wn.add_junction(name='J2', base_demand=0.008, elevation=45.0)
        self.wn.add_junction(name='J3', base_demand=0.006, elevation=40.0)
        self.wn.add_junction(name='J4', base_demand=0.007, elevation=42.0)
        self.wn.add_junction(name='J5', base_demand=0.004, elevation=38.0)
        
        self.wn.add_tank(name='T1', elevation=60.0, init_level=15.0, 
                        min_level=5.0, max_level=25.0, diameter=10.0)
        
        self.wn.add_pipe(name='P1', start_node_name='R1', end_node_name='J1',
                        length=1000.0, diameter=0.3, roughness=120)
        self.wn.add_pipe(name='P2', start_node_name='J1', end_node_name='J2',
                        length=800.0, diameter=0.25, roughness=120)
        self.wn.add_pipe(name='P3', start_node_name='J2', end_node_name='J3',
                        length=600.0, diameter=0.2, roughness=110)
        self.wn.add_pipe(name='P4', start_node_name='J1', end_node_name='J4',
                        length=700.0, diameter=0.2, roughness=115)
        self.wn.add_pipe(name='P5', start_node_name='J4', end_node_name='J5',
                        length=500.0, diameter=0.15, roughness=110)
        self.wn.add_pipe(name='P6', start_node_name='J3', end_node_name='J5',
                        length=400.0, diameter=0.15, roughness=110)
        self.wn.add_pipe(name='P7', start_node_name='J2', end_node_name='T1',
                        length=900.0, diameter=0.2, roughness=120)
        
        self.wn.options.time.duration = 0
    
    #demand update function
    def update_demand(self, node_id, new_demand_Ls):
        demand_m3s = new_demand_Ls / 1000.0  #unit conv
        junction = self.wn.get_node(node_id)
        junction.demand_timeseries_list[0].base_value = demand_m3s
        self.demands[node_id] = demand_m3s

    #update head       
    def update_reservoir_head(self, new_head):
        self.reservoir_head = new_head
        reservoir = self.wn.get_node('R1')
        reservoir.base_head = new_head
    
    #run simulation
    def solve_network(self):
        try:
            self.sim = wntr.sim.EpanetSimulator(self.wn)
            self.results = self.sim.run_sim()
        except Exception as e:
            print(f"Simulation error: {e}")

    #get node pressure in kPa 
    def get_node_pressure(self, node_id):
        if self.results is None:
            return 0
        try:
            pressure_m = self.results.node['pressure'].loc[0, node_id]
            pressure_kPa = pressure_m * 9.81  #unit conv
            return max(0, pressure_kPa)
        except:
            return 0
    
    #node head func
    def get_node_head(self, node_id):
        if self.results is None:
            return 0
        try:
            head = self.results.node['head'].loc[0, node_id]
            return head
        except:
            return 0
    
    #pipe flow rate func
    def get_pipe_flow(self, pipe_id):
        if self.results is None:
            return 0
        try:
            flow_m3s = self.results.link['flowrate'].loc[0, pipe_id]
            flow_Ls = flow_m3s * 1000.0  #unit conv
            return flow_Ls
        except:
            return 0
    
    #pipe velocity in m/s
    def get_pipe_velocity(self, pipe_id):
        if self.results is None:
            return 0
        try:
            velocity = self.results.link['velocity'].loc[0, pipe_id]
            return abs(velocity)
        except:
            return 0
    
    #timestep
    def step(self):
        self.solve_network()
        self.time_step += 1


#function to store previous data values
class DataStorage:
    
    def __init__(self, max_history=500):
        self.max_history = max_history
        self.timestamps = deque(maxlen=max_history)
        self.node_data = {node_id: {
            'pressure': deque(maxlen=max_history),
            'head': deque(maxlen=max_history)
        } for node_id in ['J1', 'J2', 'J3', 'J4', 'J5']}
        
        self.pipe_data = {pipe_id: {
            'flow': deque(maxlen=max_history),
            'velocity': deque(maxlen=max_history)
        } for pipe_id in ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7']}
        
    #add current state to history
    def add_record(self, simulator):
        self.timestamps.append(datetime.now())
        
        #noise added
        noise_pressure = np.random.normal(0, 2)
        noise_flow = np.random.normal(0, 0.2)
        
        for node_id in self.node_data.keys():
            pressure = simulator.get_node_pressure(node_id) + noise_pressure
            head = simulator.get_node_head(node_id)
            self.node_data[node_id]['pressure'].append(max(0, pressure))
            self.node_data[node_id]['head'].append(head)
        
        for pipe_id in self.pipe_data.keys():
            flow = simulator.get_pipe_flow(pipe_id) + noise_flow
            velocity = simulator.get_pipe_velocity(pipe_id)
            self.pipe_data[pipe_id]['flow'].append(flow)
            self.pipe_data[pipe_id]['velocity'].append(velocity)

#for continous simulation
class SimulationThread(threading.Thread):    
    def __init__(self, simulator, storage, update_interval=1.0):
        super().__init__(daemon=True)
        self.simulator = simulator
        self.storage = storage
        self.update_interval = update_interval
        self.running = True
        
    def run(self):
        while self.running:
            self.simulator.step()
            self.storage.add_record(self.simulator)
            time.sleep(self.update_interval)
    
    def stop(self):
        self.running = False

#beggining of dash application
simulator = WaterDistributionSimulator()
storage = DataStorage()
sim_thread = SimulationThread(simulator, storage, update_interval=1.0)
sim_thread.start()

app = dash.Dash(__name__)
app.title = "Water Distribution Digital Twin"

app.layout = html.Div([
    #header
    html.Div([
        html.H1("Digital Twin Dashboard", 
                style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 5, 'fontSize': '32px'}),
        html.P("Real-time hydraulic simulation using the WNTR (Water Network Tool for Resilience) Python package.",
               style={'textAlign': 'center', 'color': '#7f8c8d', 'marginBottom': 15, 'fontSize': '14px'})
    ]),
    
    #main dashboard - two column layout
    html.Div([
        #left column - controls and stats
        html.Div([
            #control panel
            html.Div([
                html.H3("System Controls", style={'color': '#2c3e50', 'marginBottom': 15, 'fontSize': '18px'}),
                
                html.Div([
                    html.Label("Reservoir Head (m):", style={'display': 'inline-block', 'width': '65%', 'fontSize': '13px'}),
                    html.Span(id='reservoir-value', style={'display': 'inline-block', 'width': '30%', 'textAlign': 'right', 
                             'fontWeight': 'bold', 'color': '#2c3e50', 'fontSize': '14px'}),
                ]),
                dcc.Slider(id='reservoir-slider', min=80, max=120, value=100, 
                          marks={i: str(i) for i in range(80, 121, 20)}, step=1),
                
                html.Br(),
                html.Div([
                    html.Label("Junction J2 Demand (L/s):", style={'display': 'inline-block', 'width': '65%', 'fontSize': '13px'}),
                    html.Span(id='demand-j2-value', style={'display': 'inline-block', 'width': '30%', 'textAlign': 'right', 
                             'fontWeight': 'bold', 'color': '#2c3e50', 'fontSize': '14px'}),
                ]),
                dcc.Slider(id='demand-j2-slider', min=0, max=20, value=8,
                          marks={i: str(i) for i in range(0, 21, 10)}, step=0.5),
                
                html.Br(),
                html.Div([
                    html.Label("Junction J3 Demand (L/s):", style={'display': 'inline-block', 'width': '65%', 'fontSize': '13px'}),
                    html.Span(id='demand-j3-value', style={'display': 'inline-block', 'width': '30%', 'textAlign': 'right', 
                             'fontWeight': 'bold', 'color': '#2c3e50', 'fontSize': '14px'}),
                ]),
                dcc.Slider(id='demand-j3-slider', min=0, max=20, value=6,
                          marks={i: str(i) for i in range(0, 21, 10)}, step=0.5),
                
            ], style={'backgroundColor': '#ecf0f1', 'padding': 18, 'borderRadius': 8, 'marginBottom': 20}),
            
            #stats panel
            html.Div([
                html.H3("System Statistics", style={'color': '#2c3e50', 'marginBottom': 15, 'fontSize': '18px'}),
                html.Div(id='stats-panel')
            ], style={'backgroundColor': '#ecf0f1', 'padding': 18, 'borderRadius': 8}),
            
        ], style={'width': '28%', 'display': 'inline-block', 'verticalAlign': 'top', 'paddingRight': 15}),
        
        #right column - visualizations
        html.Div([
            #network diagram
            html.Div([
                html.H3("Network Status", style={'color': '#2c3e50', 'marginBottom': 10, 'fontSize': '18px'}),
                dcc.Graph(id='network-graph', style={'height': '340px'})
            ], style={'marginBottom': 20}),
            
            #two graphs side by side
            html.Div([
                html.Div([
                    html.H3("Junction Pressures", style={'color': '#2c3e50', 'marginBottom': 8, 'fontSize': '16px'}),
                    dcc.Graph(id='pressure-graph', style={'height': '280px'})
                ], style={'width': '49%', 'display': 'inline-block'}),
                
                html.Div([
                    html.H3("Pipe Flow Rates", style={'color': '#2c3e50', 'marginBottom': 8, 'fontSize': '16px'}),
                    dcc.Graph(id='flow-graph', style={'height': '280px'})
                ], style={'width': '49%', 'display': 'inline-block', 'marginLeft': '2%'}),
            ]),
            
        ], style={'width': '70%', 'display': 'inline-block', 'verticalAlign': 'top'}),
        
    ], style={'marginTop': 10}),
    
    dcc.Interval(id='interval-component', interval=1000, n_intervals=0)
    
], style={'padding': 20, 'fontFamily': 'Arial, sans-serif', 'backgroundColor': '#f5f6fa', 'height': '100vh', 'overflow': 'hidden'})


@app.callback(
    Output('reservoir-slider', 'value'),
    Output('demand-j2-slider', 'value'),
    Output('demand-j3-slider', 'value'),
    Output('reservoir-value', 'children'),
    Output('demand-j2-value', 'children'),
    Output('demand-j3-value', 'children'),
    Input('reservoir-slider', 'value'),
    Input('demand-j2-slider', 'value'),
    Input('demand-j3-slider', 'value')
)

#update with new values
def update_simulator_parameters(reservoir_head, demand_j2, demand_j3):
    if reservoir_head is not None:
        simulator.update_reservoir_head(reservoir_head)
    if demand_j2 is not None:
        simulator.update_demand('J2', demand_j2)
    if demand_j3 is not None:
        simulator.update_demand('J3', demand_j3)
    
    simulator.solve_network()

    #formatting values
    res_display = f"{reservoir_head:.0f} m" if reservoir_head is not None else "100 m"
    j2_display = f"{demand_j2:.1f} L/s" if demand_j2 is not None else "8.0 L/s"
    j3_display = f"{demand_j3:.1f} L/s" if demand_j3 is not None else "6.0 L/s"
    
    return reservoir_head, demand_j2, demand_j3, res_display, j2_display, j3_display

@app.callback(
    Output('network-graph', 'figure'),
    Output('pressure-graph', 'figure'),
    Output('flow-graph', 'figure'),
    Output('stats-panel', 'children'),
    Input('interval-component', 'n_intervals')
)

#update live graphs
def update_graphs(n):

    node_positions = {'R1': (0, 2), 'J1': (1, 2), 'J2': (2, 2), 'J3': (3, 2),'J4': (1, 1), 'J5': (3, 1), 'T1': (2, 3)}
    
    edge_traces = []
    pipe_label_positions = {'P1': (0.5, 2.08),'P2': (1.5, 2.08),'P3': (2.5, 2.08),'P4': (0.92, 1.5),'P5': (2, 0.92),'P6': (3.08, 1.5),'P7': (1.92, 2.5)}
    
    for pipe_id in ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7']:
        pipe = simulator.wn.get_link(pipe_id)
        x0, y0 = node_positions[pipe.start_node_name]
        x1, y1 = node_positions[pipe.end_node_name]
        
        flow = simulator.get_pipe_flow(pipe_id)
        velocity = simulator.get_pipe_velocity(pipe_id)
        
        #flow rate colours
        color = 'green' if flow > 3 else 'orange' if flow > 0 else 'red'
        width = max(1, min(abs(flow)/3, 5))
        
        edge_traces.append(go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode='lines',
            line=dict(width=width, color=color),
            hoverinfo='text',
            hovertext=f"<b>{pipe_id}</b><br>Flow: {flow:.2f} L/s<br>Velocity: {velocity:.2f} m/s<br>Length: {pipe.length:.0f} m",
            showlegend=False,
            name=pipe_id
        ))
        
        label_x, label_y = pipe_label_positions[pipe_id]
        edge_traces.append(go.Scatter(
            x=[label_x],
            y=[label_y],
            mode='text',
            text=[pipe_id],
            textfont=dict(size=10, color='#2c3e50', family='Arial Black'),
            hoverinfo='skip',
            showlegend=False
        ))
    
    #node colours
    node_colours = []
    node_ids = ['R1', 'J1', 'J2', 'J3', 'J4', 'J5', 'T1']
    for node_id in node_ids:
        node = simulator.wn.get_node(node_id)
        if node.node_type == 'Reservoir':
            node_colours.append(600)
        elif node.node_type == 'Tank':
            node_colours.append(500)
        else:
            node_colours.append(simulator.get_node_pressure(node_id))
    
    node_text_positions = ['top center', 'top center', 'top left', 'top center', 
                          'bottom center', 'bottom center', 'top center']
    
    node_trace = go.Scatter(
        x=[node_positions[nid][0] for nid in node_ids],
        y=[node_positions[nid][1] for nid in node_ids],
        mode='markers+text',
        marker=dict(
            size=[35 if simulator.wn.get_node(n).node_type == 'Reservoir' 
                  else 30 if simulator.wn.get_node(n).node_type == 'Tank' 
                  else 25 for n in node_ids],
            color=node_colours,
            colorscale='Viridis',
            showscale=True,
            cmin=200,
            cmax=600,
            colorbar=dict(title="Pressure<br>(kPa)")
        ),
        text=node_ids,
        textposition=node_text_positions,
        textfont=dict(size=13, color='black', family='Arial Black'),
        texttemplate="<b>%{text}</b>",
        hovertext=[f"<b>{nid}</b><br>Pressure: {simulator.get_node_pressure(nid):.1f} kPa<br>"
                   f"Head: {simulator.get_node_head(nid):.1f} m" 
                   for nid in node_ids],
        hoverinfo='text'
    )
    
    network_fig = go.Figure(data=edge_traces + [node_trace])
    network_fig.update_layout(
        showlegend=False,
        hovermode='closest',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white',
        height=340,
        margin=dict(l=15, r=15, t=10, b=10)
    )
    
    #pressure time series
    pressure_fig = go.Figure()
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
    for idx, node_id in enumerate(['J1', 'J2', 'J3', 'J4', 'J5']):
        if len(storage.node_data[node_id]['pressure']) > 0:
            pressure_fig.add_trace(go.Scatter(
                y=list(storage.node_data[node_id]['pressure']),
                mode='lines',
                name=node_id,
                line=dict(width=2, color=colors[idx])
            ))
    
    pressure_fig.update_layout(
        xaxis_title="Time Steps",
        yaxis_title="Pressure (kPa)",
        yaxis=dict(range=[0, 700]),
        hovermode='x unified',
        plot_bgcolor='white',
        height=280,
        legend=dict(x=1.02, y=1, xanchor='left'),
        margin=dict(l=50, r=10, t=20, b=40)
    )
    
    #flow rate time series
    flow_fig = go.Figure()
    colors2 = ['#1abc9c', '#e67e22', '#8e44ad', '#c0392b', '#16a085', '#d35400', '#2980b9']
    for idx, pipe_id in enumerate(['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7']):
        if len(storage.pipe_data[pipe_id]['flow']) > 0:
            flow_fig.add_trace(go.Scatter(
                y=list(storage.pipe_data[pipe_id]['flow']),
                mode='lines',
                name=pipe_id,
                line=dict(width=2, color=colors2[idx])
            ))
    
    flow_fig.update_layout(
        xaxis_title="Time Steps",
        yaxis_title="Flow Rate (L/s)",
        hovermode='x unified',
        plot_bgcolor='white',
        height=280,
        legend=dict(x=1.02, y=1, xanchor='left'),
        margin=dict(l=50, r=10, t=20, b=40)
    )
    
    #stats panel
    total_flow = sum(abs(simulator.get_pipe_flow(p)) for p in ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7'])
    pressures = [simulator.get_node_pressure(n) for n in ['J1', 'J2', 'J3', 'J4', 'J5']]
    avg_pressure = np.mean(pressures) if pressures else 0
    total_demand = sum(simulator.demands.values()) * 1000  #conv to L/s
    
    stats = html.Div([
        html.Div([
            html.H4(f"{total_flow:.1f}", style={'color': '#3498db', 'margin': 0, 'fontSize': '24px'}),
            html.P("L/s", style={'color': '#3498db', 'margin': 0, 'fontSize': '12px', 'fontWeight': 'bold'}),
            html.P("Total Flow", style={'color': '#7f8c8d', 'margin': 0, 'fontSize': '11px', 'marginTop': 3})
        ], style={'textAlign': 'center', 'marginBottom': 15}),
        
        html.Div([
            html.H4(f"{avg_pressure:.0f}", style={'color': '#2ecc71', 'margin': 0, 'fontSize': '24px'}),
            html.P("kPa", style={'color': '#2ecc71', 'margin': 0, 'fontSize': '12px', 'fontWeight': 'bold'}),
            html.P("Avg Pressure", style={'color': '#7f8c8d', 'margin': 0, 'fontSize': '11px', 'marginTop': 3})
        ], style={'textAlign': 'center', 'marginBottom': 15}),
        
        html.Div([
            html.H4(f"{total_demand:.1f}", style={'color': '#e74c3c', 'margin': 0, 'fontSize': '24px'}),
            html.P("L/s", style={'color': '#e74c3c', 'margin': 0, 'fontSize': '12px', 'fontWeight': 'bold'}),
            html.P("Total Demand", style={'color': '#7f8c8d', 'margin': 0, 'fontSize': '11px', 'marginTop': 3})
        ], style={'textAlign': 'center'}),
    ])
    
    return network_fig, pressure_fig, flow_fig, stats

if __name__ == '__main__':
    print("WATER DISTRIBUTION SYSTEM - DIGITAL TWIN (WNTR)") 
    app.run_server(debug=False, host='127.0.0.1', port=8050)