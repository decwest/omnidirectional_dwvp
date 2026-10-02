"""Standalone, headless figures; no plotting dependency in the command kernel."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

COLORS={"vp":"#2b6cb0", "vp_min":"#56a3bf", "vp_max":"#88bdac", "dwvp":"#c44e52"}
NAMES={"iros_docking":"IROS docking", "iros_curve":"IROS curve",
       "constant_heading_corner":"Constant heading", "independent_heading_curve":"Independent heading"}
plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":9, "axes.labelsize":9,
                     "axes.titlesize":10, "legend.fontsize":8, "pdf.fonttype":42,
                     "axes.spines.top":False, "axes.spines.right":False})


def save(fig, stem):
    fig.savefig(str(stem)+".pdf", bbox_inches="tight")
    fig.savefig(str(stem)+".png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def arrows(ax, poses, color):
    points=poses[np.unique(np.linspace(0,len(poses)-1,10,dtype=int))]
    ax.quiver(points[:,0],points[:,1],np.cos(points[:,2]),np.sin(points[:,2]),
              color=color, angles="xy",scale_units="xy",scale=8,width=.004,alpha=.6)


def mechanism_figures(out, paths, results):
    fig, axes=plt.subplots(2,2,figsize=(7.2,6.3), constrained_layout=True)
    for ax,(name,path) in zip(axes.flat,paths.items()):
        ax.plot(path[:,0],path[:,1],"k--",lw=1,label="Reference")
        for method in ("vp_min","vp_max","vp","dwvp"):
            poses=results[(name,method)].arrays["poses"]
            label={"vp":"VP adaptive", "vp_min":"VP fixed min", "vp_max":"VP fixed max", "dwvp":"DWVP"}[method]
            ax.plot(poses[:,0],poses[:,1],color=COLORS[method],lw=1.3,label=label)
            if method=="dwvp": arrows(ax,poses,COLORS[method])
        arrows(ax,path,"#444444")
        ax.set(title=NAMES[name],xlabel="x [m]",ylabel="y [m]")
        ax.margins(.15)
        ax.set_aspect("equal",adjustable="box")
        ax.grid(alpha=.2)
    handles,labels=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="outside upper center",ncol=5,frameon=False)
    save(fig,out/"tracking")
    fig,axes=plt.subplots(4,2,figsize=(7.2,8.5),constrained_layout=True)
    for col,name in enumerate(("iros_docking","independent_heading_curve")):
        for method in ("vp","dwvp"):
            a=results[(name,method)].arrays
            t=a["times"][1:]
            axes[0,col].plot(t,np.linalg.norm(a["applied"][:,:2],axis=1),color=COLORS[method],label=method.upper())
            axes[1,col].plot(t,a["applied"][:,2],color=COLORS[method])
            axes[2,col].plot(t,a["direction_angle_deg"],color=COLORS[method])
            axes[3,col].plot(t,a["speed_caps"],color=COLORS[method],ls="--")
        axes[0,col].set_title(NAMES[name])
        axes[0,col].legend(frameon=False)
        for ax,label in zip(axes[:,col],("Speed [m/s]","Yaw rate [rad/s]","Direction error [deg]","Speed cap [m/s]")):
            ax.set_ylabel(label)
            ax.grid(alpha=.2)
        axes[-1,col].set_xlabel("Time [s]")
    save(fig,out/"velocity_and_direction")
    fig,axes=plt.subplots(1,3,figsize=(7.2,2.9),constrained_layout=True)
    names=list(paths)
    for ax,metric,label in zip(axes,("mean_position_error_m","mean_heading_error_deg","duration_s"),
                              ("Mean position error [m]","Mean heading error [deg]","Duration [s]")):
        for offset,method in ((-.18,"vp"),(.18,"dwvp")):
            ax.bar(np.arange(4)+offset,[results[(n,method)].metrics[metric] for n in names],.34,color=COLORS[method],label=method.upper())
        ax.set_xticks(range(4),["Dock","Curve","Const.","Indep."],rotation=25)
        ax.set_ylabel(label)
        ax.grid(axis="y",alpha=.2)
    axes[0].legend(frameon=False)
    save(fig,out/"metrics")


def obstacle_figures(out,path,results,obstacles,config):
    selected=("no_cost_no_approach","cost_no_approach","no_cost_approach","cost_approach")
    colors=("#7b8c96","#e69f00","#009e73","#c44e52")
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.5),constrained_layout=True)
    for ax,method in zip(axes,("vp","dwvp")):
        ax.plot(path[:,0],path[:,1],"k--",label="Reference")
        for label,color in zip(selected,colors):
            a=results[(label,method)].arrays
            ax.plot(a["poses"][:,0],a["poses"][:,1],color=color,label=label.replace("_"," "))
        for obstacle in obstacles:
            ax.add_patch(Circle((obstacle.x,obstacle.y),obstacle.radius,color="#777777"))
            ax.add_patch(Circle((obstacle.x,obstacle.y),obstacle.radius+config.robot_radius,fill=False,ec="#777777",ls=":"))
        ax.add_patch(Circle((0,0),config.robot_radius,fill=False,ec="black",lw=.8))
        ax.set_aspect("equal",adjustable="box")
        ax.set(title=method.upper(),xlabel="x [m]",ylabel="y [m]",xlim=(-.28,1.7),ylim=(-.7,1.28))
        ax.grid(alpha=.2)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="outside upper center",ncol=2,frameon=False)
    save(fig,out/"scene")
    fig,axes=plt.subplots(3,2,figsize=(7.2,6.1),constrained_layout=True)
    for col,method in enumerate(("vp","dwvp")):
        for label,color in zip(selected,colors):
            a=results[(label,method)].arrays
            t=a["times"][1:]
            axes[0,col].plot(t,np.linalg.norm(a["applied"][:,:2],axis=1),color=color,label=label.replace("_"," "))
            axes[1,col].plot(t,a["speed_caps"],color=color)
            axes[2,col].plot(t,a["clearance"],color=color)
        axes[0,col].set_title(method.upper())
        axes[2,col].axhline(0,color="black",ls=":",lw=.8)
        for ax,label in zip(axes[:,col],("Speed [m/s]","Speed cap [m/s]","Swept clearance [m]")):
            ax.set_ylabel(label)
            ax.grid(alpha=.2)
        axes[-1,col].set_xlabel("Time [s]")
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc="outside upper center",ncol=2,frameon=False)
    save(fig,out/"regulation")


def sweep_figures(out,rows):
    parameters=list(dict.fromkeys(row["parameter"] for row in rows))
    for parameter in parameters:
        subset=[r for r in rows if r["parameter"]==parameter]
        path_names=list(dict.fromkeys(r["path"] for r in subset))
        fig,axes=plt.subplots(3,len(path_names),figsize=(max(3.8,2.6*len(path_names)),6.8),squeeze=False,constrained_layout=True)
        cost_study=parameter.startswith("cost_")
        metrics=("mean_position_error_m","min_clearance_m" if cost_study else "mean_heading_error_deg","duration_s")
        labels=("Mean position error [m]","Min. clearance [m]" if cost_study else "Mean heading error [deg]","Duration [s]")
        xlabel={"lookahead_time":"Lookahead time [s]", "fixed_lookahead":"Fixed lookahead [m]",
                "acceleration_scale":"Acceleration multiplier", "lateral_acceleration":"Lateral acceleration [m/s²]",
                "cost_scaling_dist":"Cost distance [m]", "cost_scaling_gain":"Cost gain",
                "approach_distance":"Approach distance [m]"}[parameter]
        for col,path in enumerate(path_names):
            axes[0,col].set_title(NAMES[path])
            for method in ("vp","dwvp"):
                selected=sorted((r for r in subset if r["path"]==path and r["method"]==method),key=lambda r:r["value"])
                for ax,metric,label in zip(axes[:,col],metrics,labels):
                    ax.plot([r["value"] for r in selected],[r[metric] for r in selected],"o-",ms=3,lw=1,color=COLORS[method],label=method.upper())
                    failed=[r for r in selected if not r["success"]]
                    if failed: ax.scatter([r["value"] for r in failed],[r[metric] for r in failed],marker="x",s=55,c="black",zorder=5)
                    ax.set_ylabel(label)
                    if metric != "min_clearance_m": ax.set_ylim(bottom=0)
                    ax.grid(alpha=.2)
            axes[-1,col].set_xlabel(xlabel)
        fig.legend(*axes[0,0].get_legend_handles_labels(),loc="outside upper center",ncol=2,frameon=False)
        save(fig,out/parameter)
