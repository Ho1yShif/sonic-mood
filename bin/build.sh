#!/bin/bash

cd frontend-site
npm run build && npm run serve &
sleep 2
open http://localhost:3000

