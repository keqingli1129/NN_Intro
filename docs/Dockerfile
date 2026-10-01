#See https://aka.ms/customizecontainer to learn how to customize your debug container and how Visual Studio uses this Dockerfile to build your images for faster debugging.

# This is a MULTI-STAGE build: several temporary images ("stages") are built in sequence,
# and only the last one ("final") becomes the image you actually run. The heavy SDK used for
# compiling is thrown away, so the shipped image stays small.

# ---------------------------------------------------------------------------
# Stage 1 "base": the lightweight runtime image the app will eventually run in.
# ---------------------------------------------------------------------------

# Start from Microsoft's ASP.NET Core 8.0 runtime image (can RUN .NET web apps, cannot BUILD them)
# and name this stage "base" so later stages can refer to it.
FROM mcr.microsoft.com/dotnet/aspnet:8.0 AS base
# Switch to the non-root "app" user that ships with the .NET 8 images - safer than running as root.
USER app
# Set the working directory inside the container to /app (created if it doesn't exist).
WORKDIR /app
# Document that the app listens on port 8080 (HTTP, the .NET 8 default) ...
EXPOSE 8080
# ... and port 8081 (typically HTTPS). EXPOSE is documentation only; it doesn't publish ports -
# you still map them with `docker run -p` or `ports:` in docker-compose.
EXPOSE 8081

# ---------------------------------------------------------------------------
# Stage 2 "build": compile the source code using the full .NET SDK.
# ---------------------------------------------------------------------------

# Start a fresh stage from the .NET 8.0 SDK image (has the compiler and `dotnet build` tooling).
FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build
# Build-time variable, defaulting to "Release". Override with `--build-arg BUILD_CONFIGURATION=Debug`.
ARG BUILD_CONFIGURATION=Release
# All source code will live under /src in this stage.
WORKDIR /src
# Copy ONLY the project file first. Docker caches each step, so as long as the .csproj
# (i.e. the list of NuGet dependencies) is unchanged, the restore below is reused from cache.
COPY ["ProductsMicroService.API/ProductsMicroService.API.csproj", "ProductsMicroService.API/"]
# Download the NuGet packages the project depends on.
RUN dotnet restore "./ProductsMicroService.API/./ProductsMicroService.API.csproj"
# Now copy the rest of the source code (everything in the build context) into /src.
COPY . .
# Move into the project folder so the following commands run against it.
WORKDIR "/src/ProductsMicroService.API"
# Compile the project with the chosen configuration (Release by default) and put the output in /app/build.
RUN dotnet build "./ProductsMicroService.API.csproj" -c $BUILD_CONFIGURATION -o /app/build

# ---------------------------------------------------------------------------
# Stage 3 "publish": produce the final, deployable set of files.
# ---------------------------------------------------------------------------

# Continue on top of the "build" stage (it already has the restored packages and source).
FROM build AS publish
# ARGs don't carry over between stages, so it has to be declared again here.
ARG BUILD_CONFIGURATION=Release
# Publish the app: compile and gather the DLLs plus dependencies into /app/publish.
# /p:UseAppHost=false skips creating a platform-specific launcher executable; the app is
# started with `dotnet <name>.dll` instead (see ENTRYPOINT below).
RUN dotnet publish "./ProductsMicroService.API.csproj" -c $BUILD_CONFIGURATION -o /app/publish /p:UseAppHost=false

# ---------------------------------------------------------------------------
# Stage 4 "final": the image that actually ships.
# ---------------------------------------------------------------------------

# Start again from the small runtime "base" stage - none of the SDK or source code comes along.
FROM base AS final
# Work in /app (already set in "base"; repeated here for clarity).
WORKDIR /app
# Copy only the published output from the "publish" stage into /app.
COPY --from=publish /app/publish .
# Environment variable the app reads to find its MySQL server. "localhost" inside a container
# means the container itself, so this is normally overridden (e.g. in docker-compose) with the
# database's service name.
ENV MYSQL_HOST=localhost
# Database password, baked into the image. WARNING: anyone who can pull the image can read it
# with `docker inspect`. Prefer supplying it at run time (`-e`, docker-compose env, or secrets).
ENV MYSQL_PASSWORD=admin
# The command the container runs when it starts: launch the web API with the .NET runtime.
ENTRYPOINT ["dotnet", "ProductsMicroService.API.dll"]
