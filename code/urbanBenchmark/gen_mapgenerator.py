# -*- coding: utf-8 -*-
"""Generate a city with ProbableTrain/MapGenerator (tensor-field streets + block subdivision + building lots, LGPL-3.0; probabletrain.itch.io/city-generator) in headless Edge,
export the world-coordinate street polylines and building lots as JSON; gen_import.py mapgenerator converts them to the three-layer format.
Requires: dataset/generators/MapGenerator/dist/bundle.js patched (window.app = new Main()); pip install playwright; Microsoft Edge installed.
Usage: python gen_mapgenerator.py --repo dataset/generators/MapGenerator --out dataset/generated/mapgen_s0 [--seed 0] [--width 1440 --height 900]
Output: {out}/mapgen_raw.json {minor,major,main,coastline: [[[x,y],...],...], lots: [[[x,y],...],...], params}
Coordinates are generator world units (y down); the scale has no physical definition, gen_import uses --scale metres/unit (default 5: minor spacing dsep=20 units ~ 100 m blocks)."""
import os, json, argparse, http.server, threading, functools, socketserver

def serve(directory, port=0):
    handler=functools.partial(http.server.SimpleHTTPRequestHandler,directory=directory)
    class Q(http.server.HTTPServer): pass
    srv=Q(('127.0.0.1',port),handler); srv.timeout=1
    t=threading.Thread(target=srv.serve_forever,daemon=True); t.start(); return srv, srv.server_address[1]

JS_GENERATE="""async (seed) => {
  const gui=app.mainGui;
  // reproducible: replace Math.random by mulberry32(seed)
  let s=seed>>>0; Math.random=()=>{s+=0x6D2B79F5; let t=Math.imul(s^(s>>>15),1|s); t=t+Math.imul(t^(t>>>7),61|t)^t; return ((t^(t>>>14))>>>0)/4294967296;};
  app.tensorField.setRecommended();            // same as the recommended default tensor field of the UI
  gui.animate=false;
  await gui.generateEverything();
  return true;
}"""
JS_EXTRACT="""() => {
  const gui=app.mainGui; const V=p=>p.map(v=>[v.x,v.y]);
  const roads=r=>r.streamlines.allStreamlinesSimple.map(V);
  return {minor:roads(gui.minorRoads), major:roads(gui.majorRoads), main:roads(gui.mainRoads), coastline:roads(gui.coastline),
          lots:gui.buildings.polygonFinder.polygons.map(V), params:{minor:gui.minorParams, major:gui.majorParams, main:gui.mainParams, building:gui.buildings.buildingParams},
          zoom:app.domainController.zoom, origin:[app.domainController.origin.x,app.domainController.origin.y], screen:[app.domainController.screenDimensions.x,app.domainController.screenDimensions.y]};
}"""

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--repo',required=True); ap.add_argument('--out',required=True); ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--width',type=int,default=1440); ap.add_argument('--height',type=int,default=900); ap.add_argument('--timeout',type=int,default=600)
    a=ap.parse_args(); dist=os.path.join(os.path.abspath(a.repo),'dist'); os.makedirs(a.out,exist_ok=True)
    from playwright.sync_api import sync_playwright
    srv,port=serve(dist)
    try:
        with sync_playwright() as p:
            b=p.chromium.launch(channel='msedge',headless=True); pg=b.new_page(viewport={'width':a.width,'height':a.height})
            logs=[]; pg.on('console',lambda m: logs.append(m.text)); pg.on('pageerror',lambda e: logs.append('ERR '+str(e)))
            pg.goto(f'http://127.0.0.1:{port}/index.html'); pg.wait_for_function('() => window.app && app.mainGui',timeout=60000)
            pg.evaluate(JS_GENERATE,a.seed)
            pg.wait_for_function('() => app.mainGui.buildings.polygonFinder.polygons.length>0',timeout=a.timeout*1000)
            pg.wait_for_timeout(1500)
            d=pg.evaluate(JS_EXTRACT); b.close()
    finally: srv.shutdown()
    d['seed']=a.seed; d['source']='ProbableTrain/MapGenerator'
    json.dump(d,open(f'{a.out}/mapgen_raw.json','w'),separators=(',',':'))
    n=lambda k:(len(d[k]),sum(len(x) for x in d[k]))
    print(f"[mapgen] seed {a.seed}: roads minor {n('minor')[0]} major {n('major')[0]} main {n('main')[0]} coast {n('coastline')[0]}; lots {len(d['lots'])}; zoom {d['zoom']} -> {a.out}/mapgen_raw.json")
    if any('ERR' in l for l in logs): print('page errors:',[l for l in logs if 'ERR' in l][:3])
if __name__=='__main__': main()
