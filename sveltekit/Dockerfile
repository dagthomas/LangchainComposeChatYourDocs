FROM node:16-bullseye-slim AS BUILDER
LABEL dockerfile.baseimage="node:lts-bullseye-slim" dockerfile.description="LearningLibrary Build Container" dockerfile.stage="BUILDER"
ENV NODE_ENV development
ENV BODY_SIZE_LIMIT=0
ENV ORIGIN=http://localhost:3000
WORKDIR /usr/src/app
COPY package.json tsconfig.json ./
RUN npm i
COPY . /usr/src/app
RUN npm run build && npm prune --omit=dev

FROM gcr.io/distroless/nodejs:16

LABEL dockerfile.baseimage="gcr.io/distroless/nodejs:16" dockerfile.description="LearningLibrary Production Container" dockerfile.stage="PRODUCTION"
ENV NODE_ENV production
ENV BODY_SIZE_LIMIT=0
ENV ORIGIN=http://localhost:3000
WORKDIR /usr/src/app
COPY --from=BUILDER /usr/src/app/build ./build
COPY --from=BUILDER /usr/src/app/node_modules ./node_modules
COPY package.json .

EXPOSE 3000:3000
CMD ["build"]